"""
多输出(原始尺度 + 多步)最优参数 × 多随机种子重复训练，输出箱线图所需的逐 (horizon,target) 指标表。

读取 grid_search_multi_output.py 分步搜索得到的每架构最优超参数，固定超参数、
更换随机种子重复训练，记录每 (model, seed, dataset, horizon, target) 的指标。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.data_utils import (  # noqa: E402
    ensure_dir,
    get_raw_feature_cols,
    load_data_splits,
    prepare_multi_horizon_data,
)
from common.seed_utils import set_seed  # noqa: E402
from common.train_utils import make_loader, train_one_model  # noqa: E402
from models import HORIZONS, KAN, MODEL_DISPLAY, TARGET_TRAIN_COLS, TARGET_COLS_ORDER, build_model  # noqa: E402
from grid_search_multi_output import (  # noqa: E402
    DATA_DIR,
    FEATURE_COLS_FN,
    LOOKBACK,
    MODELS_TO_RUN,
    PARAM_COLUMNS,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
    evaluate_split,
    input_dim_for,
    output_paths,
)


RANDOM_SEEDS = [42, 2024, 2025, 3407, 12345, 7, 77, 777, 1001, 2026]
DATASETS = ["train", "val", "test"]
MAX_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 20

SMOKE = bool(os.environ.get("MULTI_SMOKE"))
if SMOKE:
    RANDOM_SEEDS = [42, 2024]
    MAX_EPOCHS = 2
    EARLY_STOPPING_PATIENCE = 2

OUTPUT_DIR = SCRIPT_DIR / "multi_seed_outputs_multi"
METRICS_PATH = OUTPUT_DIR / "metrics" / "best_params_multi_seed_metrics.csv"

RESULT_COLUMNS = [
    "model_name", "model_display", "seed", "dataset", "horizon", "target",
    *PARAM_COLUMNS, "best_epoch", "train_loss_at_best_epoch", "val_loss_at_best_epoch",
    "nRMSE", "nMAE", "NSE", "KGE",
]


def load_best_params(model_name: str) -> dict:
    path = output_paths(model_name)["best_params"]
    if not path.exists():
        raise FileNotFoundError(f"未找到 {model_name} 多输出最优参数: {path}。请先运行 grid_search_multi_output.py。")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if "LSTM_KAN" in MODELS_TO_RUN and KAN is None:
        raise ImportError("efficient-kan is required before running LSTM-KAN.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df, data_dir, date_col = load_data_splits(DATA_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV)
    feature_cols = FEATURE_COLS_FN(train_df, date_col)
    data = prepare_multi_horizon_data(train_df, val_df, test_df, feature_cols, TARGET_TRAIN_COLS, LOOKBACK, HORIZONS, date_col)
    output_dim = data["train"]["y"].shape[1]
    print(f"Data: {data_dir} | device={device} | raw features={len(feature_cols)} | "
          f"horizons={HORIZONS} | seeds={RANDOM_SEEDS} | smoke={SMOKE}", flush=True)

    all_rows = []
    for model_name in MODELS_TO_RUN:
        best_info = load_best_params(model_name)
        params = best_info["best_hyperparameters"]
        if best_info.get("feature_cols") != feature_cols:
            print(f"WARNING: {model_name} 最优参数 feature_cols 与当前 raw features 不一致。", file=sys.stderr)
        input_dim = input_dim_for(model_name, data)
        for seed in RANDOM_SEEDS:
            print(f"[{MODEL_DISPLAY[model_name]}][seed={seed}] training", flush=True)
            set_seed(seed)
            model = build_model(model_name, input_dim, params, output_dim)
            train_loader = make_loader(data["train"]["X"], data["train"]["y"], params["batch_size"], True, seed=seed)
            val_loader = make_loader(data["val"]["X"], data["val"]["y"], params["batch_size"], False, seed=seed)
            _, _, best_epoch, train_loss, val_loss = train_one_model(
                model, train_loader, val_loader, params["lr"], MAX_EPOCHS, EARLY_STOPPING_PATIENCE,
                device, weight_decay=params.get("weight_decay", 0.0),
            )
            for dataset in DATASETS:
                per, _, _ = evaluate_split(model, data[dataset], data["y_scaler"], params["batch_size"], device, output_dim)
                for (h_idx, name), m in per.items():
                    all_rows.append({
                        "model_name": model_name, "model_display": MODEL_DISPLAY[model_name],
                        "seed": seed, "dataset": dataset, "horizon": HORIZONS[h_idx], "target": name,
                        **{c: params.get(c) for c in PARAM_COLUMNS},
                        "best_epoch": best_epoch, "train_loss_at_best_epoch": train_loss,
                        "val_loss_at_best_epoch": val_loss, **m,
                    })

    ensure_dir(METRICS_PATH.parent)
    pd.DataFrame(all_rows).reindex(columns=RESULT_COLUMNS).to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    print(f"指标表格已保存: {METRICS_PATH}", flush=True)


if __name__ == "__main__":
    main()
