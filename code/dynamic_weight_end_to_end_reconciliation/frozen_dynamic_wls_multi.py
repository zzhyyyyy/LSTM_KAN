"""
多输出(原始尺度 + 多步)基座的 frozen DynamicWLS 端到端协调。

基座 = 一个多输出 LSTM-KAN（冻结），一次输出 t+1/t+2/t+3 × 5 目标。
层次一致性在每个 horizon 上分别成立，所以可微 WLS 协调层把 (N,H) 展平成批量、
对每个样本-horizon 的 5 维向量各做一次加权最小二乘协调。原始尺度：反标准化后不 expm1。

复用单输出 frozen_dynamic_wls 的：DynamicWeightMLP、汇总矩阵、is_better_trial、MLP_GRID
及数值常量(WEIGHT_EPS/SOLVE_RIDGE/GRAD_CLIP_NORM/CONSISTENCY_LOSS_WEIGHT)。

列序桥接 RECON_FROM_MODEL == [4,0,1,2,3]；列序与 y_scaler 统计量必须同步重排成 top-first。
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
for _p in (PROJECT_DIR, SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from base_model.common.data_utils import (  # noqa: E402
    ensure_dir,
    get_raw_feature_cols,
    inverse_targets,
    load_data_splits,
    prepare_multi_horizon_data,
)
from base_model.common.metrics_utils import compute_metrics  # noqa: E402
from base_model.common.seed_utils import set_seed  # noqa: E402
from base_model.common.train_utils import save_loss_history  # noqa: E402
from base_model.multi_output_model.models import HORIZONS, KAN, TARGET_COLS_ORDER, build_model  # noqa: E402

from frozen_dynamic_wls import (  # noqa: E402
    ALL_TARGETS,
    CONSISTENCY_LOSS_WEIGHT,
    EPS,
    GRAD_CLIP_NORM,
    MLP_GRID,
    SOLVE_RIDGE,
    DynamicWeightMLP,
    build_summing_matrix,
    is_better_trial,
    iter_grid,
    json_safe,
    metric_text,
    params_text,
)


BEST_OUTPUT_DIR = PROJECT_DIR / "base_model" / "multi_output_model" / "grid_search_outputs_multi"
OUTPUT_DIR = SCRIPT_DIR / "frozen_dynamic_wls_multi_outputs"
MODEL_NAME = "LSTM_KAN"
MODEL_STEM = "lstm_kan"
SEED = 42

DATA_DIR = PROJECT_DIR / "base_model" / "data"
TRAIN_CSV, VAL_CSV, TEST_CSV = "train_model_input.csv", "val_model_input.csv", "test_model_input.csv"
DATASETS = ["train", "val", "test"]

MAX_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 20
SMOKE = bool(os.environ.get("MULTI_SMOKE"))
if SMOKE:
    MAX_EPOCHS = 2
    EARLY_STOPPING_PATIENCE = 2

RECON_FROM_MODEL = [TARGET_COLS_ORDER.index(name) for name in ALL_TARGETS]
assert sorted(RECON_FROM_MODEL) == list(range(len(ALL_TARGETS)))

METRIC_COLUMNS = ["model", "horizon", "target", "dataset", "method", "nRMSE", "nMAE", "NSE", "KGE"]
CONSISTENCY_COLUMNS = ["model", "dataset", "horizon", "D1_mean_abs", "D2_mean_abs", "consistency_improvement_mean_pct"]


class DynamicHierMultiHorizonModel(nn.Module):
    """冻结多输出多步基座 + 动态权重 WLS；对每个样本-horizon 的 5 维向量分别协调。"""

    def __init__(self, base_model, means_top, scales_top, mlp_params, recon_from_model, n_horizons, n_targets):
        super().__init__()
        self.base = base_model
        self.n_h, self.n_k = n_horizons, n_targets
        self.weight_mlp = DynamicWeightMLP(n_targets, mlp_params["hidden_dim"], mlp_params["num_layers"], mlp_params["dropout"])
        self.register_buffer("target_means", torch.tensor(means_top, dtype=torch.float32))
        self.register_buffer("target_scales", torch.tensor(scales_top, dtype=torch.float32))
        self.register_buffer("summing_matrix", torch.tensor(build_summing_matrix(), dtype=torch.float32))
        self.register_buffer("recon_from_model", torch.tensor(recon_from_model, dtype=torch.long))

    def forward(self, x):
        n = x.shape[0]
        base_scaled = self.base(x).view(n, self.n_h, self.n_k)  # 模型原生列序(标准化)
        base_scaled = base_scaled.index_select(2, self.recon_from_model)  # top-first
        # 原始尺度：反标准化但不 expm1
        base_pred = base_scaled * self.target_scales.view(1, 1, -1) + self.target_means.view(1, 1, -1)
        flat = base_pred.reshape(n * self.n_h, self.n_k)
        weights = self.weight_mlp(flat)
        reconciled_flat, _ = self.reconcile(flat, weights)
        reconciled = reconciled_flat.view(n, self.n_h, self.n_k)
        return {"base_pred": base_pred, "reconciled": reconciled, "weights": weights.view(n, self.n_h, self.n_k)}

    def reconcile(self, base_pred, weights):
        s = self.summing_matrix
        w_inv = 1.0 / weights
        weighted_s = s.unsqueeze(0) * w_inv.unsqueeze(-1)
        middle = torch.matmul(s.T.unsqueeze(0), weighted_s)
        eye = torch.eye(s.shape[1], device=base_pred.device, dtype=base_pred.dtype)
        middle = middle + SOLVE_RIDGE * eye.unsqueeze(0)
        right = torch.matmul(s.T.unsqueeze(0), (w_inv * base_pred).unsqueeze(-1))
        bottom = torch.linalg.solve(middle, right).squeeze(-1)
        return torch.matmul(bottom, s.T), bottom


def load_best_info() -> dict:
    path = BEST_OUTPUT_DIR / "best_params" / f"best_params_{MODEL_STEM}.json"
    if not path.exists():
        raise FileNotFoundError(f"未找到多输出 LSTM-KAN 最优参数: {path}。请先运行 grid_search_multi_output.py。")
    return json.loads(path.read_text(encoding="utf-8"))


def true_top_first(data, dataset, n_h, n_k, y_scaler) -> np.ndarray:
    """标准化目标 -> 原始 -> top-first，返回 (N,H,K)。"""
    std = data[dataset]["y"].reshape(-1, n_h, n_k)
    raw = np.stack([inverse_targets(std[:, h, :], y_scaler) for h in range(n_h)], axis=1)  # model order
    return raw[:, :, RECON_FROM_MODEL].astype(np.float32)  # top-first


def make_loader(data, dataset, batch_size, shuffle, n_h, n_k, y_scaler, seed=SEED) -> DataLoader:
    x = torch.from_numpy(data[dataset]["X"]).float()
    y = torch.from_numpy(true_top_first(data, dataset, n_h, n_k, y_scaler)).float()  # (N,H,K)
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, generator=gen)


def build_dynamic_model(best_info, data, mlp_params, n_h, n_k):
    params = best_info["best_hyperparameters"]
    base = build_model(MODEL_NAME, data["train"]["X"].shape[-1], params, data["train"]["y"].shape[1])
    state_path = BEST_OUTPUT_DIR / "best_models" / f"best_{MODEL_STEM}.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"未找到多输出 LSTM-KAN 最优权重: {state_path}")
    base.load_state_dict(torch.load(state_path, map_location="cpu"))
    means = data["y_scaler"].mean_[RECON_FROM_MODEL]
    scales = data["y_scaler"].scale_[RECON_FROM_MODEL]
    model = DynamicHierMultiHorizonModel(base, means, scales, mlp_params, RECON_FROM_MODEL, n_h, n_k)
    for p in model.base.parameters():
        p.requires_grad = False
    return model


def batch_loss(output, y_true, criterion):
    pred_loss = criterion(output["reconciled"], y_true)
    resid = output["reconciled"][..., 0] - output["reconciled"][..., 1:].sum(dim=-1)
    return pred_loss + CONSISTENCY_LOSS_WEIGHT * torch.mean(resid**2)


def train_dynamic(model, train_loader, val_loader, optimizer, device, max_epochs, patience):
    criterion = nn.MSELoss()
    model.to(device)
    best_state, best_val, best_epoch, train_at_best, wait = None, float("inf"), 0, float("inf"), 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        tsum, tn = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = batch_loss(model(xb), yb, criterion)
            loss.backward()
            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            tsum += loss.item() * yb.size(0)
            tn += yb.size(0)
        train_loss = tsum / max(tn, 1)
        model.eval()
        vsum, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vsum += batch_loss(model(xb), yb, criterion).item() * yb.size(0)
                vn += yb.size(0)
        val_loss = vsum / max(vn, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-12:
            best_val, best_epoch, train_at_best = val_loss, epoch, train_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(copy.deepcopy(best_state))
    return history, best_state, best_epoch, train_at_best, best_val


@torch.no_grad()
def predict(model, loader, device):
    model.to(device)
    model.eval()
    store = {"true": [], "base": [], "reconciled": []}
    for xb, yb in loader:
        out = model(xb.to(device))
        store["true"].append(yb.numpy())
        store["base"].append(out["base_pred"].cpu().numpy())
        store["reconciled"].append(out["reconciled"].cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in store.items()}  # each (N,H,K) top-first


def safe_pct(before, after):
    if not np.isfinite(before) or abs(before) <= EPS:
        return np.nan
    return float((before - after) / before * 100.0)


def evaluate_split(preds, dataset, horizons):
    """返回 (metric_rows, consistency_rows)。preds: {true,base,reconciled} (N,H,K) top-first。"""
    metric_rows, cons_rows = [], []
    for h_idx, horizon in enumerate(horizons):
        base_h = preds["base"][:, h_idx, :]
        recon_h = preds["reconciled"][:, h_idx, :]
        true_h = preds["true"][:, h_idx, :]
        for ti, target in enumerate(ALL_TARGETS):
            for method, arr in [("Base", base_h), ("DynamicWLS", recon_h)]:
                m = compute_metrics(true_h[:, ti], arr[:, ti])
                metric_rows.append({"model": "LSTM-KAN", "horizon": horizon, "target": target, "dataset": dataset,
                                    "method": method, "nRMSE": m["nRMSE"], "nMAE": m["nMAE"], "NSE": m["NSE"], "KGE": m["KGE"]})
        d1 = np.abs(base_h[:, 0] - base_h[:, 1:].sum(axis=1))
        d2 = np.abs(recon_h[:, 0] - recon_h[:, 1:].sum(axis=1))
        d1m, d2m = float(np.nanmean(d1)), float(np.nanmean(d2))
        cons_rows.append({"model": "LSTM-KAN", "dataset": dataset, "horizon": horizon,
                          "D1_mean_abs": d1m, "D2_mean_abs": d2m, "consistency_improvement_mean_pct": safe_pct(d1m, d2m)})
    return metric_rows, cons_rows


def val_summary(metric_rows, cons_rows):
    dyn = [r for r in metric_rows if r["dataset"] == "val" and r["method"] == "DynamicWLS"]
    vcons = [r for r in cons_rows if r["dataset"] == "val"]
    return {
        "validation_mean_nRMSE": float(np.nanmean([r["nRMSE"] for r in dyn])),
        "validation_D1_mean_abs": float(np.nanmean([r["D1_mean_abs"] for r in vcons])),
        "validation_D2_mean_abs": float(np.nanmean([r["D2_mean_abs"] for r in vcons])),
        "validation_consistency_improvement_mean_pct": float(np.nanmean([r["consistency_improvement_mean_pct"] for r in vcons])),
    }


def ensure_output_dirs():
    for sub in ["search", "loss_history", "best_params", "best_models", "metrics", "consistency"]:
        ensure_dir(OUTPUT_DIR / sub)


def main() -> None:
    set_seed(SEED)
    if KAN is None:
        raise ImportError("未检测到 efficient-kan，请先安装后再运行 DynamicWLS。")
    ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df, actual_data_dir, date_col = load_data_splits(DATA_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV)
    get_raw_feature_cols(train_df, date_col)
    best_info = load_best_info()
    params = best_info["best_hyperparameters"]
    horizons = best_info["horizons"]
    data = prepare_multi_horizon_data(train_df, val_df, test_df, best_info["feature_cols"], best_info["target_cols"],
                                      params["lookback"], horizons, date_col)
    n_h, n_k = len(horizons), len(TARGET_COLS_ORDER)
    batch_size = int(params["batch_size"])
    y_scaler = data["y_scaler"]

    grid = iter_grid(MLP_GRID)
    if SMOKE:
        grid = grid[:1]
    print(f"Data: {actual_data_dir} | device={device} | base=multi-output LSTM-KAN (frozen) | "
          f"horizons={horizons} | MLP trials={len(grid)} | smoke={SMOKE}", flush=True)
    print(f"RECON_FROM_MODEL: {RECON_FROM_MODEL}", flush=True)

    best_summary, best_bundle, best_mlp = None, None, None
    search_rows = []
    for idx, mlp_params in enumerate(grid, start=1):
        trial_id = f"trial_{idx:04d}"
        set_seed(SEED)
        model = build_dynamic_model(best_info, data, mlp_params, n_h, n_k)
        train_loader = make_loader(data, "train", batch_size, True, n_h, n_k, y_scaler)
        val_loader = make_loader(data, "val", batch_size, False, n_h, n_k, y_scaler)
        optimizer = torch.optim.Adam([p for p in model.weight_mlp.parameters() if p.requires_grad], lr=mlp_params["lr"])
        history, best_state, best_epoch, train_loss, val_loss = train_dynamic(
            model, train_loader, val_loader, optimizer, device, MAX_EPOCHS, EARLY_STOPPING_PATIENCE
        )
        save_loss_history(history, OUTPUT_DIR / "loss_history" / MODEL_NAME / f"{trial_id}_loss.csv")

        metric_rows, cons_rows = [], []
        for dataset in DATASETS:
            preds = predict(model, make_loader(data, dataset, batch_size, False, n_h, n_k, y_scaler), device)
            mr, cr = evaluate_split(preds, dataset, horizons)
            metric_rows.extend(mr)
            cons_rows.extend(cr)
        summary = val_summary(metric_rows, cons_rows)
        search_rows.append({"trial_id": trial_id, **{f"mlp_{k}": v for k, v in mlp_params.items()},
                            "best_epoch": best_epoch, "val_loss": val_loss, **summary})

        if is_better_trial(summary, best_summary):
            best_summary, best_bundle, best_mlp = summary, (metric_rows, cons_rows), mlp_params
            torch.save(best_state, OUTPUT_DIR / "best_models" / f"best_dynamic_weight_{MODEL_STEM}.pt")
            (OUTPUT_DIR / "best_params" / f"best_dynamic_weight_{MODEL_STEM}.json").write_text(
                json.dumps(json_safe({
                    "model_name": MODEL_NAME, "targets_top_first": ALL_TARGETS, "horizons": horizons,
                    "recon_from_model": RECON_FROM_MODEL, "mlp_hyperparameters": mlp_params,
                    "joint_batch_size": batch_size, "base_best_info": best_info, "best_epoch": best_epoch, **summary,
                }), ensure_ascii=False, indent=2), encoding="utf-8")
        mark = " [Best Updated]" if best_mlp is mlp_params else ""
        print(f"[DWLS-multi] Trial {idx}/{len(grid)} | MLP {params_text(mlp_params)} | "
              f"val mean nRMSE={metric_text(summary['validation_mean_nRMSE'])} | "
              f"D1={metric_text(summary['validation_D1_mean_abs'])} | D2={metric_text(summary['validation_D2_mean_abs'])}{mark}", flush=True)

    if best_bundle is not None:
        metric_rows, cons_rows = best_bundle
        pd.DataFrame(metric_rows, columns=METRIC_COLUMNS).to_csv(
            OUTPUT_DIR / "metrics" / f"{MODEL_STEM}_dynamic_weight_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(cons_rows, columns=CONSISTENCY_COLUMNS).to_csv(
            OUTPUT_DIR / "consistency" / f"{MODEL_STEM}_consistency_by_horizon.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(search_rows).to_csv(OUTPUT_DIR / "search" / f"{MODEL_STEM}_mlp_search.csv", index=False, encoding="utf-8-sig")
    print(f"Output directory: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
