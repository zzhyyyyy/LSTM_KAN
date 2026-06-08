"""
多输出基础模型超参数优化（LSTM-FC / LSTM-MLP / LSTM-KAN）。

三个关键设定（按用户与研究进展文档确认）：
1. 原始尺度：输入 13 个原始特征（不含 Algae_Sum、不做 log），目标 5 个原始列；
   预测只反标准化、不 expm1（inverse_targets）。
2. 多步预测：一个模型一次输出 t+1/t+2/t+3 × 5 目标 = 15 个值（direct multi-horizon），
   按 (horizon, target) 分别评估；选优用跨全部 (h,k) 的平均 val nRMSE。
3. 分步优化（坐标下降）：从基线出发，按"结构/学习率优先"的顺序逐个超参数扫描选优、
   固定后再调下一个——而非穷举全网格（贴合文档"先逐一调关键参数、再缩小范围"）。

烟雾测试：MULTI_SMOKE=1 → 每个超参数只留基线一个候选 + 2 epoch。
"""

from __future__ import annotations

import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.data_utils import (  # noqa: E402
    append_csv_row,
    ensure_dir,
    get_log_feature_cols,
    get_raw_feature_cols,
    inverse_log_targets,
    inverse_targets,
    load_data_splits,
    prepare_multi_horizon_data,
)
from common.metrics_utils import compute_metrics  # noqa: E402
from common.seed_utils import set_seed  # noqa: E402
from common.train_utils import make_loader, predict_scaled_multi, save_loss_history, train_one_model  # noqa: E402
from models import (  # noqa: E402
    HORIZONS,
    KAN,
    LOG_INPUTS,
    MODEL_DISPLAY,
    MODEL_STEM,
    SEARCH_METHOD,
    TARGET_COLS_ORDER,
    TARGET_TRAIN_COLS,
    USE_LOG,
    build_model,
)

# 尺度选择器：USE_LOG 控制目标 log1p+expm1；LOG_INPUTS 控制输入是否 log。协调/指标都在原始尺度。
INVERSE_FN = inverse_log_targets if USE_LOG else inverse_targets
FEATURE_COLS_FN = get_log_feature_cols if LOG_INPUTS else get_raw_feature_cols


DATA_DIR = BASE_DIR / "data"
TRAIN_CSV = "train_model_input.csv"
VAL_CSV = "val_model_input.csv"
TEST_CSV = "test_model_input.csv"

OUTPUT_DIR = SCRIPT_DIR / "grid_search_outputs_multi"
SEED = 42
LOOKBACK = 30
MAX_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 20

SMOKE = bool(os.environ.get("MULTI_SMOKE"))
if SMOKE:
    MAX_EPOCHS = 2
    EARLY_STOPPING_PATIENCE = 2

MODELS_TO_RUN = ["LSTM_FC", "LSTM_MLP", "LSTM_KAN"]

# 分步优化第一阶段固定的基线配置（用户指定 BASE_*；每个值都必须落在 PARAM_ORDER 候选里，保证坐标下降单调不增）。
BASELINES = {
    "LSTM_FC": {"num_layers": 2, "hidden_dim": 256, "lr": 0.002, "dropout": 0.1, "batch_size": 32, "weight_decay": 1e-4},
    "LSTM_MLP": {"num_layers": 2, "hidden_dim": 256, "mlp_num_layers": 1, "mlp_hidden_dim": 64, "lr": 0.002, "dropout": 0.1, "batch_size": 32, "weight_decay": 1e-4},
    "LSTM_KAN": {"num_layers": 2, "hidden_dim": 256, "kan_hidden_dim": 64, "grid_size": 5, "spline_order": 2, "lr": 0.002, "dropout": 0.1, "batch_size": 32, "weight_decay": 1e-4},
}

# 坐标下降调参顺序与候选范围（SEARCH_PLAN：扩大 lr/hidden_dim/num_layers/weight_decay，其余沿用原范围）。
PARAM_ORDER = {
    "LSTM_FC": [
        ("num_layers", [1, 2, 3, 4]),
        ("hidden_dim", [64, 128, 256, 384]),
        ("lr", [0.0005, 0.001, 0.002, 0.003]),
        ("dropout", [0.0, 0.1, 0.2]),
        ("batch_size", [32, 64]),
        ("weight_decay", [0.0, 1e-5, 1e-4]),
    ],
    "LSTM_MLP": [
        ("num_layers", [1, 2, 3, 4]),
        ("hidden_dim", [64, 128, 256, 384]),
        ("mlp_num_layers", [1, 2]),
        ("mlp_hidden_dim", [32, 64, 128]),
        ("lr", [0.0005, 0.001, 0.002, 0.003]),
        ("dropout", [0.0, 0.1, 0.2]),
        ("batch_size", [32, 64]),
        ("weight_decay", [0.0, 1e-5, 1e-4]),
    ],
    "LSTM_KAN": [
        ("num_layers", [1, 2, 3, 4]),
        ("hidden_dim", [64, 128, 256, 384]),
        ("kan_hidden_dim", [32, 64, 128]),
        ("grid_size", [3, 5, 7]),
        ("spline_order", [2, 3]),
        ("lr", [0.0005, 0.001, 0.002, 0.003]),
        ("dropout", [0.0, 0.1, 0.2]),
        ("batch_size", [32, 64]),
        ("weight_decay", [0.0, 1e-5, 1e-4]),
    ],
}

# 小范围网格（SEARCH_METHOD="grid"）：只网格"层数/隐藏维/学习率"等关键超参，其余固定在 BASELINES。
# 每个架构 2×2×2=8 组，控制规模。其余超参（dropout/batch/weight_decay/mlp_*/kan_hidden/spline）取基线值。
GRID_SMALL = {
    "LSTM_FC": {"num_layers": [1, 2], "hidden_dim": [128, 256], "lr": [0.001, 0.002]},
    "LSTM_MLP": {"num_layers": [1, 2], "hidden_dim": [128, 256], "lr": [0.001, 0.002]},
    "LSTM_KAN": {"num_layers": [1, 2], "grid_size": [3, 5], "lr": [0.001, 0.002]},
}


def grid_combos(model_name: str) -> list[dict]:
    """返回小网格的完整参数组合列表（网格参数 × 其余取基线）。SMOKE 时只留基线一组。"""
    base = BASELINES[model_name]
    if SMOKE:
        return [dict(base)]
    grid = GRID_SMALL[model_name]
    keys = list(grid.keys())
    return [{**base, **dict(zip(keys, vals))} for vals in product(*(grid[k] for k in keys))]


PARAM_COLUMNS = [
    "lookback", "num_layers", "hidden_dim", "dropout", "lr", "batch_size", "weight_decay",
    "mlp_num_layers", "mlp_hidden_dim", "kan_hidden_dim", "grid_size", "spline_order",
]
PER_HORIZON_NRMSE_COLS = [f"val_nRMSE_h{h}" for h in HORIZONS]
PER_TARGET_NRMSE_COLS = [f"val_nRMSE_{name}" for name in TARGET_COLS_ORDER]
RESULT_COLUMNS = [
    "model_name", "trial_id", "tuned_param", *PARAM_COLUMNS,
    "best_epoch", "train_loss_at_best_epoch", "val_loss_at_best_epoch",
    "validation_mean_nRMSE", "validation_mean_nMAE", "validation_mean_NSE", "validation_mean_KGE",
    *PER_HORIZON_NRMSE_COLS, *PER_TARGET_NRMSE_COLS,
]
METRIC_COLUMNS = ["model_name", "horizon", "target", "dataset", "nRMSE", "nMAE", "NSE", "KGE"]


def shrink_for_smoke(param_order: list, baseline: dict) -> list:
    return [(p, [baseline[p]]) for p, _ in param_order]


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if np.isfinite(value) else None
    return obj


def metric_text(value: float) -> str:
    return "nan" if value is None or not np.isfinite(value) else f"{value:.6f}"


def output_paths(model_name: str) -> dict[str, Path]:
    stem = MODEL_STEM[model_name]
    return {
        "search": OUTPUT_DIR / "search" / f"{stem}_search_results.csv",
        "loss_dir": OUTPUT_DIR / "loss_history" / model_name,
        "best_params": OUTPUT_DIR / "best_params" / f"best_params_{stem}.json",
        "best_model": OUTPUT_DIR / "best_models" / f"best_{stem}.pt",
        "pred_dir": OUTPUT_DIR / "predictions" / model_name,
    }


def ensure_output_dirs() -> None:
    for subdir in ["search", "loss_history", "best_params", "best_models", "metrics", "predictions"]:
        ensure_dir(OUTPUT_DIR / subdir)


def input_dim_for(model_name: str, data: dict) -> int:
    """LSTM 家族用特征数；纯 KAN 见 kan_full_grid_search_multi.py。"""
    return data["train"]["X"].shape[-1]


def evaluate_split(model, split, y_scaler, batch_size, device, output_dim):
    """返回 ((h_idx,name)->metrics, pred_raw (N,H,K), true_raw (N,H,K))，原始尺度。"""
    n_h, n_k = len(HORIZONS), len(TARGET_COLS_ORDER)
    pred = predict_scaled_multi(model, split["X"], batch_size, device, output_dim).reshape(-1, n_h, n_k)
    true = split["y"].reshape(-1, n_h, n_k)
    pred_raw = np.stack([INVERSE_FN(pred[:, h, :], y_scaler) for h in range(n_h)], axis=1)
    true_raw = np.stack([INVERSE_FN(true[:, h, :], y_scaler) for h in range(n_h)], axis=1)
    per = {}
    for h in range(n_h):
        for k, name in enumerate(TARGET_COLS_ORDER):
            per[(h, name)] = compute_metrics(true_raw[:, h, k], pred_raw[:, h, k])
    return per, pred_raw, true_raw


def agg_mean(per: dict, key: str) -> float:
    return float(np.nanmean([m[key] for m in per.values()]))


def per_horizon_mean_nrmse(per: dict) -> list[float]:
    out = []
    for h in range(len(HORIZONS)):
        out.append(float(np.nanmean([per[(h, name)]["nRMSE"] for name in TARGET_COLS_ORDER])))
    return out


def per_target_mean_nrmse(per: dict) -> list[float]:
    out = []
    for name in TARGET_COLS_ORDER:
        out.append(float(np.nanmean([per[(h, name)]["nRMSE"] for h in range(len(HORIZONS))])))
    return out


def save_predictions_long(path: Path, dates, pred_raw, true_raw) -> None:
    """长表：date,horizon,target,true_value,pred_value。dates 形状 (N,H)。"""
    ensure_dir(path.parent)
    frames = []
    for h_idx, horizon in enumerate(HORIZONS):
        date_h = pd.to_datetime(dates[:, h_idx]).strftime("%Y-%m-%d") if dates is not None else None
        for k, name in enumerate(TARGET_COLS_ORDER):
            frame = pd.DataFrame(
                {"horizon": horizon, "target": name,
                 "true_value": true_raw[:, h_idx, k], "pred_value": pred_raw[:, h_idx, k]}
            )
            if date_h is not None:
                frame.insert(0, "date", date_h)
            frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")


def coordinate_descent(model_name, data, feature_cols, output_dim, device):
    """坐标下降分步搜索：逐个超参数扫描，返回最优配置与其训练得到的 best_state。"""
    paths = output_paths(model_name)
    baseline = BASELINES[model_name]
    param_order = PARAM_ORDER[model_name]
    if SMOKE:
        param_order = shrink_for_smoke(param_order, baseline)
    input_dim = input_dim_for(model_name, data)
    cache: dict = {}
    counter = {"n": 0}

    def key(params: dict) -> tuple:
        return tuple(sorted(params.items()))

    def evaluate(params: dict, tuned_param: str) -> dict:
        k = key(params)
        if k in cache:
            return cache[k]
        counter["n"] += 1
        trial_id = f"trial_{counter['n']:04d}"
        set_seed(SEED)
        model = build_model(model_name, input_dim, params, output_dim)
        train_loader = make_loader(data["train"]["X"], data["train"]["y"], params["batch_size"], True, seed=SEED)
        val_loader = make_loader(data["val"]["X"], data["val"]["y"], params["batch_size"], False, seed=SEED)
        history, best_state, best_epoch, train_loss, val_loss = train_one_model(
            model, train_loader, val_loader, params["lr"], MAX_EPOCHS, EARLY_STOPPING_PATIENCE,
            device, weight_decay=params.get("weight_decay", 0.0),
        )
        save_loss_history(history, paths["loss_dir"] / f"{trial_id}_loss.csv")
        per, _, _ = evaluate_split(model, data["val"], data["y_scaler"], params["batch_size"], device, output_dim)
        mean_nrmse = agg_mean(per, "nRMSE")
        ph = per_horizon_mean_nrmse(per)
        pt = per_target_mean_nrmse(per)
        row = {
            "model_name": model_name, "trial_id": trial_id, "tuned_param": tuned_param,
            **{c: params.get(c) for c in PARAM_COLUMNS},
            "best_epoch": best_epoch, "train_loss_at_best_epoch": train_loss, "val_loss_at_best_epoch": val_loss,
            "validation_mean_nRMSE": mean_nrmse, "validation_mean_nMAE": agg_mean(per, "nMAE"),
            "validation_mean_NSE": agg_mean(per, "NSE"), "validation_mean_KGE": agg_mean(per, "KGE"),
            **{f"val_nRMSE_h{HORIZONS[i]}": ph[i] for i in range(len(HORIZONS))},
            **{f"val_nRMSE_{TARGET_COLS_ORDER[i]}": pt[i] for i in range(len(TARGET_COLS_ORDER))},
        }
        append_csv_row(paths["search"], row, RESULT_COLUMNS)
        append_csv_row(OUTPUT_DIR / "search" / "all_models_search_results.csv", row, RESULT_COLUMNS)
        result = {"mean_nrmse": mean_nrmse, "val_loss": val_loss, "best_state": best_state,
                  "best_epoch": best_epoch, "params": dict(params), "per": per}
        cache[k] = result
        return result

    # === 网格搜索（小范围穷举）===
    if SEARCH_METHOD == "grid":
        combos = grid_combos(model_name)
        best = None
        for combo in combos:
            res = evaluate({"lookback": LOOKBACK, **combo}, "grid")
            better = best is None or res["mean_nrmse"] < best["mean_nrmse"] - 1e-9 or (
                abs(res["mean_nrmse"] - best["mean_nrmse"]) <= 1e-9 and res["val_loss"] < best["val_loss"])
            if better:
                best = res
            print(f"[{MODEL_DISPLAY[model_name]}] grid {len(cache)}/{len(combos)} | "
                  f"val mean nRMSE={metric_text(res['mean_nrmse'])} | best={metric_text(best['mean_nrmse'])}", flush=True)
        return best

    # === 分步坐标下降 ===
    current = {"lookback": LOOKBACK, **baseline}
    base_res = evaluate(current, "baseline")
    best = base_res
    print(f"[{MODEL_DISPLAY[model_name]}] baseline val mean nRMSE={metric_text(base_res['mean_nrmse'])}", flush=True)

    for pname, candidates in param_order:
        stage_best_value = current[pname]
        stage_best = cache[key(current)]
        for v in candidates:
            trial = {**current, pname: v}
            res = evaluate(trial, pname)
            improved = res["mean_nrmse"] < stage_best["mean_nrmse"] - 1e-9 or (
                abs(res["mean_nrmse"] - stage_best["mean_nrmse"]) <= 1e-9 and res["val_loss"] < stage_best["val_loss"]
            )
            if improved:
                stage_best_value, stage_best = v, res
        current[pname] = stage_best_value
        if stage_best["mean_nrmse"] < best["mean_nrmse"]:
            best = stage_best
        print(f"[{MODEL_DISPLAY[model_name]}] tuned {pname} -> {stage_best_value} | "
              f"val mean nRMSE={metric_text(stage_best['mean_nrmse'])}", flush=True)

    final = cache[key(current)]
    return final


def run_model(model_name, data, feature_cols, output_dim, device) -> list[dict]:
    paths = output_paths(model_name)
    final = coordinate_descent(model_name, data, feature_cols, output_dim, device)
    params = final["params"]

    ensure_dir(paths["best_model"].parent)
    torch.save(final["best_state"], paths["best_model"])
    val_per = final["per"]
    payload = {
        "model_name": model_name, "model_display": MODEL_DISPLAY[model_name],
        "target_names": TARGET_COLS_ORDER, "target_cols": TARGET_TRAIN_COLS,
        "feature_cols": feature_cols, "horizons": HORIZONS,
        "best_hyperparameters": params, "best_epoch": final["best_epoch"],
        "best_validation_val_loss": final["val_loss"],
        "best_validation_mean_nRMSE": final["mean_nrmse"],
        "best_validation_mean_nMAE": agg_mean(val_per, "nMAE"),
        "best_validation_mean_NSE": agg_mean(val_per, "NSE"),
        "best_validation_mean_KGE": agg_mean(val_per, "KGE"),
        "best_validation_per_horizon_nRMSE": {
            f"h{HORIZONS[i]}": per_horizon_mean_nrmse(val_per)[i] for i in range(len(HORIZONS))
        },
        "best_validation_per_target_nRMSE": {
            TARGET_COLS_ORDER[i]: per_target_mean_nrmse(val_per)[i] for i in range(len(TARGET_COLS_ORDER))
        },
    }
    ensure_dir(paths["best_params"].parent)
    paths["best_params"].write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    # 用最优配置重训一次以拿到模型并落盘 train/val/test 预测与逐 (h,target) 指标。
    input_dim = input_dim_for(model_name, data)
    set_seed(SEED)
    model = build_model(model_name, input_dim, params, output_dim)
    model.load_state_dict(torch.load(paths["best_model"], map_location="cpu"))
    metric_rows = []
    for dataset in ["train", "val", "test"]:
        per, pred_raw, true_raw = evaluate_split(model, data[dataset], data["y_scaler"], params["batch_size"], device, output_dim)
        save_predictions_long(paths["pred_dir"] / f"{MODEL_STEM[model_name]}_best_predictions_{dataset}.csv",
                              data[dataset]["dates"], pred_raw, true_raw)
        for (h_idx, name), m in per.items():
            metric_rows.append({"model_name": model_name, "horizon": HORIZONS[h_idx], "target": name,
                                "dataset": dataset, **m})
    return metric_rows


def main() -> None:
    set_seed(SEED)
    ensure_output_dirs()
    if "LSTM_KAN" in MODELS_TO_RUN and KAN is None:
        raise ImportError("efficient-kan is required before running LSTM-KAN.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df, actual_data_dir, date_col = load_data_splits(DATA_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV)
    feature_cols = FEATURE_COLS_FN(train_df, date_col)
    data = prepare_multi_horizon_data(train_df, val_df, test_df, feature_cols, TARGET_TRAIN_COLS, LOOKBACK, HORIZONS, date_col)
    output_dim = data["train"]["y"].shape[1]  # H*K
    print(f"Data: {actual_data_dir} | device={device} | raw features={len(feature_cols)} | "
          f"targets={len(TARGET_COLS_ORDER)} | horizons={HORIZONS} | output_dim={output_dim} | smoke={SMOKE}", flush=True)
    print(f"Targets: {TARGET_COLS_ORDER} | strategy=coordinate-descent (staged)", flush=True)

    all_metric_rows = []
    for model_name in MODELS_TO_RUN:
        print(f"\n=== {MODEL_DISPLAY[model_name]} (staged search) ===", flush=True)
        all_metric_rows.extend(run_model(model_name, data, feature_cols, output_dim, device))

    if all_metric_rows:
        metrics_path = OUTPUT_DIR / "metrics" / "best_metrics_all_models.csv"
        pd.DataFrame(all_metric_rows, columns=METRIC_COLUMNS).to_csv(metrics_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved best metrics: {metrics_path}", flush=True)
    print(f"Output directory: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
