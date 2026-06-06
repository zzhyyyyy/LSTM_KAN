"""
单输出·多步 基座的 frozen DynamicWLS 端到端协调。

基座 = 5 个“每目标一个”的单输出 LSTM-KAN（各冻结，各输出该目标的 t+1/t+2/t+3）。
动态权重 MLP + 可微 WLS 协调层对每个样本-horizon 的 5 维向量分别协调。原始尺度（不 expm1）。

复用 frozen_dynamic_wls_multi 的训练/预测/评估/一致性逻辑（它们只依赖 model(x) 的输出字典），
仅替换为“5 分支拼接”的基座模型类与加载逻辑。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
for _p in (PROJECT_DIR, SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from base_model.common.data_utils import (  # noqa: E402
    ensure_dir,
    get_raw_feature_cols,
    load_data_splits,
    prepare_multi_horizon_data,
)
from base_model.common.seed_utils import set_seed  # noqa: E402
from base_model.common.train_utils import save_loss_history  # noqa: E402
from base_model.multi_output_model.models import HORIZONS, KAN, TARGET_COLS, build_model  # noqa: E402

from frozen_dynamic_wls import (  # noqa: E402
    MLP_GRID, SOLVE_RIDGE, DynamicWeightMLP, build_summing_matrix,
    is_better_trial, iter_grid, json_safe, metric_text, params_text,
)
from frozen_dynamic_wls_multi import (  # noqa: E402
    ALL_TARGETS, CONSISTENCY_COLUMNS, METRIC_COLUMNS, RECON_FROM_MODEL,
    evaluate_split, make_loader, train_dynamic, predict, val_summary,
)


BEST_DIR = PROJECT_DIR / "base_model" / "single_output_mh_model" / "grid_search_outputs_single_mh"
OUTPUT_DIR = SCRIPT_DIR / "frozen_dynamic_wls_single_mh_outputs"
MODEL_STEM = "lstm_kan"
SEED = 42
DATA_DIR = PROJECT_DIR / "base_model" / "data"
TRAIN_CSV, VAL_CSV, TEST_CSV = "train_model_input.csv", "val_model_input.csv", "test_model_input.csv"

MAX_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 20
SMOKE = bool(os.environ.get("MULTI_SMOKE"))
if SMOKE:
    MAX_EPOCHS = 2
    EARLY_STOPPING_PATIENCE = 2


class DynamicHierSingleMHModel(nn.Module):
    """5 个单输出分支(top-first 顺序) + 动态权重 WLS；每样本-horizon 的 5 维向量分别协调。"""

    def __init__(self, branches_top_first, means_top, scales_top, mlp_params, n_h, n_k):
        super().__init__()
        self.branches = nn.ModuleList(branches_top_first)  # 顺序 = ALL_TARGETS (top-first)
        self.n_h, self.n_k = n_h, n_k
        self.weight_mlp = DynamicWeightMLP(n_k, mlp_params["hidden_dim"], mlp_params["num_layers"], mlp_params["dropout"])
        self.register_buffer("target_means", torch.tensor(means_top, dtype=torch.float32))
        self.register_buffer("target_scales", torch.tensor(scales_top, dtype=torch.float32))
        self.register_buffer("summing_matrix", torch.tensor(build_summing_matrix(), dtype=torch.float32))

    def forward(self, x):
        scaled = [b(x) for b in self.branches]  # each (N,H)，已是 top-first 顺序
        base_scaled = torch.stack(scaled, dim=2)  # (N,H,K)
        base_pred = base_scaled * self.target_scales.view(1, 1, -1) + self.target_means.view(1, 1, -1)
        flat = base_pred.reshape(-1, self.n_k)
        weights = self.weight_mlp(flat)
        recon_flat, _ = self.reconcile(flat, weights)
        return {"base_pred": base_pred, "reconciled": recon_flat.view(base_pred.shape), "weights": weights.view(base_pred.shape)}

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


def build_dynamic_model(data, mlp_params, n_h, n_k):
    branches = []
    for t in ALL_TARGETS:  # top-first 顺序加载
        bp = BEST_DIR / "best_params" / t / f"best_params_{MODEL_STEM}.json"
        if not bp.exists():
            raise FileNotFoundError(f"未找到 {t} 单输出 LSTM-KAN 最优参数: {bp}。请先运行 single_output_mh_model/grid_search_single_mh.py。")
        info = json.loads(bp.read_text(encoding="utf-8"))
        b = build_model("LSTM_KAN", data["train"]["X"].shape[-1], info["best_hyperparameters"], n_h)
        b.load_state_dict(torch.load(BEST_DIR / "best_models" / t / f"best_{MODEL_STEM}.pt", map_location="cpu"))
        branches.append(b)
    # 每目标的标准化统计量 == 联合 y_scaler 对应列（StandardScaler 按列独立），按 top-first 重排取用。
    means = data["y_scaler"].mean_[RECON_FROM_MODEL]
    scales = data["y_scaler"].scale_[RECON_FROM_MODEL]
    model = DynamicHierSingleMHModel(branches, means, scales, mlp_params, n_h, n_k)
    for b in model.branches:
        for p in b.parameters():
            p.requires_grad = False
    return model


def ensure_output_dirs():
    for sub in ["search", "loss_history", "best_params", "best_models", "metrics", "consistency"]:
        ensure_dir(OUTPUT_DIR / sub)


def main():
    set_seed(SEED)
    if KAN is None:
        raise ImportError("未检测到 efficient-kan，请先安装后再运行 DynamicWLS。")
    ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df, actual_data_dir, date_col = load_data_splits(DATA_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV)
    feature_cols = get_raw_feature_cols(train_df, date_col)
    # 联合准备一次：拿到共享 X、日期、K 列 y_scaler（其各列统计量与各单输出分支一致）。
    data = prepare_multi_horizon_data(train_df, val_df, test_df, feature_cols, TARGET_COLS, 30, HORIZONS, date_col)
    n_h, n_k = len(HORIZONS), len(ALL_TARGETS)
    y_scaler = data["y_scaler"]
    # batch_size 取各分支的最小值，保证都能整除
    bss = []
    for t in ALL_TARGETS:
        info = json.loads((BEST_DIR / "best_params" / t / f"best_params_{MODEL_STEM}.json").read_text(encoding="utf-8"))
        bss.append(int(info["best_hyperparameters"]["batch_size"]))
    batch_size = min(bss)

    grid = iter_grid(MLP_GRID)
    if SMOKE:
        grid = grid[:1]
    print(f"Data: {actual_data_dir} | device={device} | base=5×单输出 LSTM-KAN(frozen) | "
          f"horizons={HORIZONS} | MLP trials={len(grid)} | smoke={SMOKE}", flush=True)

    best_summary, best_bundle, best_mlp = None, None, None
    search_rows = []
    for idx, mlp_params in enumerate(grid, start=1):
        trial_id = f"trial_{idx:04d}"
        set_seed(SEED)
        model = build_dynamic_model(data, mlp_params, n_h, n_k)
        tl = make_loader(data, "train", batch_size, True, n_h, n_k, y_scaler)
        vl = make_loader(data, "val", batch_size, False, n_h, n_k, y_scaler)
        optimizer = torch.optim.Adam([p for p in model.weight_mlp.parameters() if p.requires_grad], lr=mlp_params["lr"])
        history, best_state, best_epoch, train_loss, val_loss = train_dynamic(model, tl, vl, optimizer, device, MAX_EPOCHS, EARLY_STOPPING_PATIENCE)
        save_loss_history(history, OUTPUT_DIR / "loss_history" / "LSTM_KAN" / f"{trial_id}_loss.csv")

        metric_rows, cons_rows = [], []
        for dataset in ["train", "val", "test"]:
            preds = predict(model, make_loader(data, dataset, batch_size, False, n_h, n_k, y_scaler), device)
            mr, cr = evaluate_split(preds, dataset, HORIZONS)
            metric_rows.extend(mr)
            cons_rows.extend(cr)
        summary = val_summary(metric_rows, cons_rows)
        search_rows.append({"trial_id": trial_id, **{f"mlp_{k}": v for k, v in mlp_params.items()},
                            "best_epoch": best_epoch, "val_loss": val_loss, **summary})

        if is_better_trial(summary, best_summary):
            best_summary, best_bundle, best_mlp = summary, (metric_rows, cons_rows), mlp_params
            torch.save(best_state, OUTPUT_DIR / "best_models" / f"best_dynamic_weight_{MODEL_STEM}.pt")
            (OUTPUT_DIR / "best_params" / f"best_dynamic_weight_{MODEL_STEM}.json").write_text(
                json.dumps(json_safe({"model_name": "LSTM_KAN", "base": "5×single-output", "targets_top_first": ALL_TARGETS,
                                      "horizons": HORIZONS, "mlp_hyperparameters": mlp_params, "joint_batch_size": batch_size,
                                      "best_epoch": best_epoch, **summary}), ensure_ascii=False, indent=2), encoding="utf-8")
        mark = " [Best Updated]" if best_mlp is mlp_params else ""
        print(f"[DWLS-single] Trial {idx}/{len(grid)} | MLP {params_text(mlp_params)} | "
              f"val mean nRMSE={metric_text(summary['validation_mean_nRMSE'])} | "
              f"D1={metric_text(summary['validation_D1_mean_abs'])} | D2={metric_text(summary['validation_D2_mean_abs'])}{mark}", flush=True)

    if best_bundle is not None:
        metric_rows, cons_rows = best_bundle
        pd.DataFrame(metric_rows, columns=METRIC_COLUMNS).to_csv(OUTPUT_DIR / "metrics" / f"{MODEL_STEM}_dynamic_weight_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(cons_rows, columns=CONSISTENCY_COLUMNS).to_csv(OUTPUT_DIR / "consistency" / f"{MODEL_STEM}_consistency_by_horizon.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(search_rows).to_csv(OUTPUT_DIR / "search" / f"{MODEL_STEM}_mlp_search.csv", index=False, encoding="utf-8-sig")
    print(f"Output directory: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
