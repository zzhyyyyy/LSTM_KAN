"""
单输出·多步 LSTM-KAN 的层次协调（BU/TD/OLS/WLS/MinT）。

基础预测来自“每个目标各一个”单输出 LSTM-KAN（每个一次输出该目标的 t+1/t+2/t+3）。
逐 horizon：把 5 个目标在该 horizon 的预测拼成 top-first (K,N) 矩阵后做协调。
协调数学完全复用单输出(log)模块；本脚本为原始尺度（反变换只反标准化）。
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
    inverse_targets,
    load_data_splits,
    prepare_multi_horizon_data,
)
from base_model.common.metrics_utils import compute_metrics  # noqa: E402
from base_model.common.seed_utils import set_seed  # noqa: E402
from base_model.common.train_utils import predict_scaled_multi  # noqa: E402
from base_model.multi_output_model.models import HORIZONS, KAN, TARGET_MAP_RAW, build_model  # noqa: E402

from reconcile_single_lstm_kan_best_params import (  # noqa: E402
    ALL_TARGETS,
    METHODS,
    MINT_SHRINKAGE,
    estimate_mint_shrink_cov,
    reconcile_predictions,
    td_proportions,
)


BEST_DIR = PROJECT_DIR / "base_model" / "single_output_mh_model" / "grid_search_outputs_single_mh"
OUTPUT_DIR = SCRIPT_DIR / "reconciliation_outputs_single_mh"
MODEL_STEM = "lstm_kan"
SEED = 42

DATA_DIR = PROJECT_DIR / "base_model" / "data"
TRAIN_CSV, VAL_CSV, TEST_CSV = "train_model_input.csv", "val_model_input.csv", "test_model_input.csv"

PREDICTION_METHOD_COLUMNS = {"Base": "base_pred", "BU": "BU_pred", "TD": "TD_pred",
                             "OLS": "OLS_pred", "WLS": "WLS_pred", "MinT": "MinT_pred"}
PREDICTION_COLUMNS = ["date", "horizon", "sample_index", "target", "y_true", *PREDICTION_METHOD_COLUMNS.values()]
METRIC_COLUMNS = ["model", "horizon", "target", "method", "nRMSE", "nMAE", "NSE", "KGE"]


def load_target_outputs(target, train_df, val_df, test_df, date_col, device):
    bp = BEST_DIR / "best_params" / target / f"best_params_{MODEL_STEM}.json"
    if not bp.exists():
        raise FileNotFoundError(f"未找到 {target} 单输出 LSTM-KAN 最优参数: {bp}。请先运行 single_output_mh_model/grid_search_single_mh.py。")
    info = json.loads(bp.read_text(encoding="utf-8"))
    params = info["best_hyperparameters"]
    data = prepare_multi_horizon_data(train_df, val_df, test_df, info["feature_cols"],
                                      [info["target_col"]], params["lookback"], info["horizons"], date_col)
    model = build_model("LSTM_KAN", data["train"]["X"].shape[-1], params, len(HORIZONS))
    sp = BEST_DIR / "best_models" / target / f"best_{MODEL_STEM}.pt"
    model.load_state_dict(torch.load(sp, map_location="cpu"))
    out = {}
    for split in ["val", "test"]:
        pred = predict_scaled_multi(model, data[split]["X"], params["batch_size"], device, len(HORIZONS))  # (N,H)
        true = data[split]["y"]
        pred_raw = np.stack([inverse_targets(pred[:, [h]], data["y_scaler"])[:, 0] for h in range(len(HORIZONS))], axis=1)
        true_raw = np.stack([inverse_targets(true[:, [h]], data["y_scaler"])[:, 0] for h in range(len(HORIZONS))], axis=1)
        out[split] = {"pred": pred_raw, "true": true_raw, "dates": data[split]["dates"]}  # (N,H)
    return out


def main():
    set_seed(SEED)
    if KAN is None:
        raise ImportError("未检测到 efficient-kan，请先安装后再运行 LSTM-KAN 协调。")
    ensure_dir(OUTPUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df, actual_data_dir, date_col = load_data_splits(DATA_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV)
    get_raw_feature_cols(train_df, date_col)
    proportions = td_proportions(train_df)

    # 逐目标加载单输出模型的预测（top-first 顺序）
    outs = {t: load_target_outputs(t, train_df, val_df, test_df, date_col, device) for t in ALL_TARGETS}
    n_h = len(HORIZONS)
    test_dates = outs[ALL_TARGETS[0]]["test"]["dates"]

    summary_rows, pred_rows = [], []
    for h_idx, horizon in enumerate(HORIZONS):
        val_base = np.stack([outs[t]["val"]["pred"][:, h_idx] for t in ALL_TARGETS], axis=0)   # (K,N)
        val_true = np.stack([outs[t]["val"]["true"][:, h_idx] for t in ALL_TARGETS], axis=0)
        test_base = np.stack([outs[t]["test"]["pred"][:, h_idx] for t in ALL_TARGETS], axis=0)
        test_true = np.stack([outs[t]["test"]["true"][:, h_idx] for t in ALL_TARGETS], axis=0)

        mint_cov = estimate_mint_shrink_cov(val_true - val_base, shrinkage=MINT_SHRINKAGE)
        reconciled = reconcile_predictions(test_base, proportions, mint_cov)

        date_h = pd.to_datetime(test_dates[:, h_idx]).strftime("%Y-%m-%d") if test_dates is not None else None
        sample_index = np.arange(test_true.shape[1])
        for ti, target in enumerate(ALL_TARGETS):
            rv = {"horizon": horizon, "sample_index": sample_index, "target": target, "y_true": test_true[ti]}
            for method, col in PREDICTION_METHOD_COLUMNS.items():
                rv[col] = reconciled[method][ti]
            frame = pd.DataFrame(rv)
            frame.insert(0, "date", date_h if date_h is not None else np.nan)
            pred_rows.append(frame[PREDICTION_COLUMNS])
            for method in METHODS:
                m = compute_metrics(test_true[ti], reconciled[method][ti])
                summary_rows.append({"model": "LSTM-KAN", "horizon": horizon, "target": target, "method": method,
                                     "nRMSE": m["nRMSE"], "nMAE": m["nMAE"], "NSE": m["NSE"], "KGE": m["KGE"]})

    ensure_dir(OUTPUT_DIR / "summary")
    ensure_dir(OUTPUT_DIR / "predictions")
    sp = OUTPUT_DIR / "summary" / "reconciliation_metrics.csv"
    pd.DataFrame(summary_rows, columns=METRIC_COLUMNS).to_csv(sp, index=False, encoding="utf-8-sig")
    pd.concat(pred_rows, ignore_index=True).to_csv(OUTPUT_DIR / "predictions" / "reconciled_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"Data: {actual_data_dir} | device={device} | base=5×单输出 LSTM-KAN | horizons={HORIZONS}")
    print(f"TD proportions: {dict(zip(ALL_TARGETS[1:], proportions.round(6)))}")
    print(f"Methods: {', '.join(METHODS)} | per-horizon reconciliation done")
    print(f"Summary metrics: {sp}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
