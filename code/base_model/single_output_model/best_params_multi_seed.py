from __future__ import annotations

import json
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
    get_clean_feature_cols,
    load_data_splits,
    prepare_single_target_data,
)
from common.seed_utils import set_seed  # noqa: E402
from common.train_utils import make_loader, train_one_model  # noqa: E402
from grid_search_single_output import (  # noqa: E402
    DATA_DIR,
    EARLY_STOPPING_PATIENCE,
    HORIZON,
    KAN,
    MAX_EPOCHS,
    MODEL_DISPLAY,
    MODEL_STEM,
    MODELS_TO_RUN,
    PARAM_COLUMNS,
    TARGETS_TO_RUN,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
    build_model,
    evaluate_on_split,
)


# 多随机种子重复训练，只输出后续箱线图所需的指标表。
RANDOM_SEEDS = [42, 2024, 2025, 3407, 12345, 7, 77, 777, 1001, 2026]
DATASETS = ["train", "val", "test"]
GRID_SEARCH_OUTPUT_DIR = SCRIPT_DIR / "grid_search_outputs"
OUTPUT_DIR = SCRIPT_DIR / "multi_seed_outputs"
METRICS_PATH = OUTPUT_DIR / "metrics" / "best_params_multi_seed_metrics.csv"

RESULT_COLUMNS = [
    "target",
    "target_col",
    "model_name",
    "model_display",
    "seed",
    "dataset",
    *PARAM_COLUMNS,
    "best_epoch",
    "train_loss_at_best_epoch",
    "val_loss_at_best_epoch",
    "nRMSE",
    "nMAE",
    "NSE",
    "KGE",
]


def load_best_params(target: str, model_name: str) -> tuple[str, list[str], dict]:
    """读取网格搜索阶段保存的最佳参数。"""
    path = (
        GRID_SEARCH_OUTPUT_DIR
        / "best_params"
        / target
        / f"best_params_{MODEL_STEM[model_name]}.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"未找到最佳参数文件: {path}")

    info = json.loads(path.read_text(encoding="utf-8"))
    return info["target_col"], info["feature_cols"], info["best_hyperparameters"]


def train_with_seed(
    target: str,
    target_col: str,
    model_name: str,
    params: dict,
    data: dict,
    input_dim: int,
    seed: int,
    device: torch.device,
) -> list[dict]:
    """使用指定随机种子训练一次，并计算 train/val/test 指标。"""
    set_seed(seed)
    model = build_model(model_name, input_dim, params)
    train_loader = make_loader(
        data["train"]["X"], data["train"]["y"], params["batch_size"], True, seed=seed
    )
    val_loader = make_loader(
        data["val"]["X"], data["val"]["y"], params["batch_size"], False, seed=seed
    )

    _, _, best_epoch, train_loss, val_loss = train_one_model(
        model,
        train_loader,
        val_loader,
        params["lr"],
        MAX_EPOCHS,
        EARLY_STOPPING_PATIENCE,
        device,
        weight_decay=params.get("weight_decay", 0.0),
    )

    rows = []
    for dataset in DATASETS:
        _, _, metrics = evaluate_on_split(
            model, data[dataset], data["y_scaler"], params["batch_size"], device
        )
        rows.append(
            {
                "target": target,
                "target_col": target_col,
                "model_name": model_name,
                "model_display": MODEL_DISPLAY[model_name],
                "seed": seed,
                "dataset": dataset,
                **params,
                "best_epoch": best_epoch,
                "train_loss_at_best_epoch": train_loss,
                "val_loss_at_best_epoch": val_loss,
                **metrics,
            }
        )
    return rows


def main() -> None:
    if "LSTM_KAN" in MODELS_TO_RUN and KAN is None:
        raise ImportError("未检测到 efficient-kan，请先安装 efficient-kan 后再运行 LSTM-KAN。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df, data_dir, date_col = load_data_splits(
        DATA_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV
    )
    clean_feature_cols = get_clean_feature_cols(train_df, date_col)

    all_rows = []
    print(
        f"Data: {data_dir} | device={device} | "
        f"clean features={len(clean_feature_cols)} | seeds={RANDOM_SEEDS}"
    )
    for target in TARGETS_TO_RUN:
        for model_name in MODELS_TO_RUN:
            target_col, feature_cols, params = load_best_params(target, model_name)
            missing_from_clean = [col for col in feature_cols if col not in clean_feature_cols]
            if missing_from_clean:
                raise ValueError(
                    f"{target}/{MODEL_DISPLAY[model_name]} 的网格搜索 feature_cols 含有当前 "
                    f"clean data 中不存在的列: {missing_from_clean}。"
                    "请确认 grid_search_outputs 来自 clean data 网格搜索。"
                )
            if feature_cols != clean_feature_cols:
                extra_clean = [col for col in clean_feature_cols if col not in feature_cols]
                print(
                    f"WARNING: {target}/{MODEL_DISPLAY[model_name]} 的网格搜索 feature_cols 与当前 "
                    "get_clean_feature_cols 不完全一致，将按网格搜索保存的列顺序构造输入以匹配模型权重。"
                    f" extra_clean={extra_clean}",
                    file=sys.stderr,
                )
            data = prepare_single_target_data(
                train_df,
                val_df,
                test_df,
                feature_cols,
                target_col,
                params["lookback"],
                HORIZON,
                date_col,
            )
            input_dim = data["train"]["X"].shape[-1]

            for seed in RANDOM_SEEDS:
                print(f"[{target}][{MODEL_DISPLAY[model_name]}][seed={seed}] training")
                all_rows.extend(
                    train_with_seed(
                        target, target_col, model_name, params, data, input_dim, seed, device
                    )
                )

    ensure_dir(METRICS_PATH.parent)
    pd.DataFrame(all_rows).reindex(columns=RESULT_COLUMNS).to_csv(
        METRICS_PATH, index=False, encoding="utf-8-sig"
    )
    print(f"指标表格已保存: {METRICS_PATH}")


if __name__ == "__main__":
    main()
