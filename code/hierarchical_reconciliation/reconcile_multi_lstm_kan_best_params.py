"""
多输出(原始尺度 + 多步) LSTM-KAN 的层次协调（BU/TD/OLS/WLS/MinT）。

基础预测来自“一个”多输出 LSTM-KAN，一次输出 t+1/t+2/t+3 × 5 目标。
层次一致性(总藻=四藻之和)在“每个 horizon 上分别成立”，所以对每个 horizon 各做一次协调。
协调数学(汇总矩阵/投影/MinT 协方差/各方法)完全复用单输出模块；原始尺度，反变换只反标准化。

列序桥接：模型原生 [Green,Cyano,Diatoms,Crypto,Algae_Sum] -> 协调 top-first
ALL_TARGETS=[Algae_Sum,Green,Cyano,Diatoms,Crypto]，RECON_FROM_MODEL == [4,0,1,2,3]。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
for _p in (PROJECT_DIR, SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from base_model.common.data_utils import (  # noqa: E402
    ensure_dir,
    get_raw_feature_cols,
    inverse_log_targets,
    inverse_targets,
    load_data_splits,
    prepare_multi_horizon_data,
)
from base_model.common.metrics_utils import compute_metrics  # noqa: E402
from base_model.common.seed_utils import set_seed  # noqa: E402
from base_model.common.train_utils import predict_scaled_multi  # noqa: E402
from base_model.multi_output_model.models import HORIZONS, KAN, TARGET_COLS_ORDER, USE_LOG, build_model  # noqa: E402

# 方案A：目标 log1p 时反变换用 expm1；协调在原始尺度。
INVERSE_FN = inverse_log_targets if USE_LOG else inverse_targets

from reconcile_single_lstm_kan_best_params import (  # noqa: E402
    ALL_TARGETS,
    METHODS,
    MINT_SHRINKAGE,
    estimate_mint_shrink_cov,
    reconcile_predictions,
    td_proportions,
)


BEST_OUTPUT_DIR = PROJECT_DIR / "base_model" / "multi_output_model" / "grid_search_outputs_multi"
OUTPUT_DIR = SCRIPT_DIR / "reconciliation_outputs_multi"
MODEL_NAME = "LSTM_KAN"
MODEL_STEM = "lstm_kan"
SEED = 42

DATA_DIR = PROJECT_DIR / "base_model" / "data"
TRAIN_CSV, VAL_CSV, TEST_CSV = "train_model_input.csv", "val_model_input.csv", "test_model_input.csv"

RECON_FROM_MODEL = [TARGET_COLS_ORDER.index(name) for name in ALL_TARGETS]
assert set(TARGET_COLS_ORDER) == set(ALL_TARGETS)
assert sorted(RECON_FROM_MODEL) == list(range(len(ALL_TARGETS)))

PREDICTION_METHOD_COLUMNS = {
    "Base": "base_pred", "BU": "BU_pred", "TD": "TD_pred",
    "OLS": "OLS_pred", "WLS": "WLS_pred", "MinT": "MinT_pred",
}
PREDICTION_COLUMNS = ["date", "horizon", "sample_index", "target", "y_true", *PREDICTION_METHOD_COLUMNS.values()]
METRIC_COLUMNS = ["model", "horizon", "target", "method", "nRMSE", "nMAE", "NSE", "KGE"]


def load_best_info() -> dict:
    path = BEST_OUTPUT_DIR / "best_params" / f"best_params_{MODEL_STEM}.json"
    if not path.exists():
        raise FileNotFoundError(f"未找到多输出 LSTM-KAN 最优参数: {path}。请先运行 grid_search_multi_output.py。")
    info = json.loads(path.read_text(encoding="utf-8"))
    for field in ["feature_cols", "target_cols", "best_hyperparameters", "horizons"]:
        if field not in info:
            raise KeyError(f"{path} 缺少必要字段: {field}")
    return info


def to_top_first(mat_model_order: np.ndarray) -> np.ndarray:
    """(N,K) 模型原生列序 -> (K,N) 协调 top-first。"""
    return mat_model_order[:, RECON_FROM_MODEL].T


def main() -> None:
    set_seed(SEED)
    if KAN is None:
        raise ImportError("未检测到 efficient-kan，请先安装后再运行 LSTM-KAN 协调。")
    ensure_dir(OUTPUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df, actual_data_dir, date_col = load_data_splits(DATA_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV)
    get_raw_feature_cols(train_df, date_col)  # 列存在性检查
    best_info = load_best_info()
    feature_cols, target_cols = best_info["feature_cols"], best_info["target_cols"]
    params = best_info["best_hyperparameters"]
    horizons = best_info["horizons"]
    n_h, n_k = len(horizons), len(TARGET_COLS_ORDER)

    data = prepare_multi_horizon_data(train_df, val_df, test_df, feature_cols, target_cols, params["lookback"], horizons, date_col)
    output_dim = data["train"]["y"].shape[1]
    model = build_model(MODEL_NAME, data["train"]["X"].shape[-1], params, output_dim)
    state_path = BEST_OUTPUT_DIR / "best_models" / f"best_{MODEL_STEM}.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"未找到多输出 LSTM-KAN 最优权重: {state_path}")
    model.load_state_dict(torch.load(state_path, map_location="cpu"))

    def split_raw(split):
        pred = predict_scaled_multi(model, split["X"], params["batch_size"], device, output_dim).reshape(-1, n_h, n_k)
        true = split["y"].reshape(-1, n_h, n_k)
        pred_raw = np.stack([INVERSE_FN(pred[:, h, :], data["y_scaler"]) for h in range(n_h)], axis=1)
        true_raw = np.stack([INVERSE_FN(true[:, h, :], data["y_scaler"]) for h in range(n_h)], axis=1)
        return pred_raw, true_raw  # (N,H,K)

    val_pred, val_true = split_raw(data["val"])
    test_pred, test_true = split_raw(data["test"])
    proportions = td_proportions(train_df)
    test_dates = data["test"]["dates"]  # (N,H)

    summary_rows, pred_rows = [], []
    for h_idx, horizon in enumerate(horizons):
        val_base_top = to_top_first(val_pred[:, h_idx, :])
        val_true_top = to_top_first(val_true[:, h_idx, :])
        test_base_top = to_top_first(test_pred[:, h_idx, :])
        test_true_top = to_top_first(test_true[:, h_idx, :])

        mint_cov = estimate_mint_shrink_cov(val_true_top - val_base_top, shrinkage=MINT_SHRINKAGE)
        reconciled = reconcile_predictions(test_base_top, proportions, mint_cov)  # method -> (K,N)

        date_h = pd.to_datetime(test_dates[:, h_idx]).strftime("%Y-%m-%d") if test_dates is not None else None
        sample_index = np.arange(test_true_top.shape[1])
        for ti, target in enumerate(ALL_TARGETS):
            row_values = {"horizon": horizon, "sample_index": sample_index, "target": target,
                          "y_true": test_true_top[ti]}
            for method, col in PREDICTION_METHOD_COLUMNS.items():
                row_values[col] = reconciled[method][ti]
            frame = pd.DataFrame(row_values)
            frame.insert(0, "date", date_h if date_h is not None else np.nan)
            pred_rows.append(frame[PREDICTION_COLUMNS])
            for method in METHODS:
                m = compute_metrics(test_true_top[ti], reconciled[method][ti])
                summary_rows.append({"model": "LSTM-KAN", "horizon": horizon, "target": target, "method": method,
                                     "nRMSE": m["nRMSE"], "nMAE": m["nMAE"], "NSE": m["NSE"], "KGE": m["KGE"]})

    ensure_dir(OUTPUT_DIR / "summary")
    ensure_dir(OUTPUT_DIR / "predictions")
    summary_path = OUTPUT_DIR / "summary" / "reconciliation_metrics.csv"
    pd.DataFrame(summary_rows, columns=METRIC_COLUMNS).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.concat(pred_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "predictions" / "reconciled_predictions.csv", index=False, encoding="utf-8-sig"
    )

    print(f"Data: {actual_data_dir} | device={device} | base=multi-output LSTM-KAN | horizons={horizons}")
    print(f"RECON_FROM_MODEL (model->top-first): {RECON_FROM_MODEL}")
    print(f"TD proportions from train: {dict(zip(ALL_TARGETS[1:], proportions.round(6)))}")
    print(f"Methods: {', '.join(METHODS)} | per-horizon reconciliation done")
    print(f"Summary metrics: {summary_path}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
