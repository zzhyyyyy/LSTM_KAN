from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from common.data_utils import (  # noqa: E402
    append_csv_row,
    ensure_dir,
    get_clean_feature_cols,
    inverse_log_target,
    load_completed_trials,
    load_data_splits,
    prepare_single_target_data,
)
from common.metrics_utils import compute_metrics  # noqa: E402
from common.seed_utils import set_seed  # noqa: E402
from common.train_utils import make_loader, predict_scaled, save_loss_history, train_one_model  # noqa: E402

try:
    from efficient_kan import KAN
except ImportError:
    KAN = None


# =========================
# 路径与运行配置
# =========================
DATA_DIR = BASE_DIR / "data"
TRAIN_CSV = "train_model_input.csv"
VAL_CSV = "val_model_input.csv"
TEST_CSV = "test_model_input.csv"

OUTPUT_DIR = SCRIPT_DIR / "grid_search_outputs"
SEED = 42
LOOKBACK = 30
HORIZON = 1
MAX_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 20
SKIP_EXISTING = True
PROTECTED_TARGETS = set()

TARGET_MAP = {
    "Green_Algae": "log_Green_Algae",
    "Cyanobacteria": "log_Cyanobacteria",
    "Diatoms": "log_Diatoms",
    "Cryptophyta": "log_Cryptophyta",
    "Algae_Sum": "log_Algae_Sum",
}

TARGETS_TO_RUN = [
    "Green_Algae",
    "Cyanobacteria",
    "Diatoms",
    "Cryptophyta",
    "Algae_Sum",
]
FORCE_RERUN_TARGETS = set(TARGETS_TO_RUN)

MODELS_TO_RUN = [
    "LSTM_FC",
    "LSTM_MLP",
    "LSTM_KAN",
]

GRID_SPACES = {
    "LSTM_FC": {
        "hidden_dim": [256],
        "num_layers": [1, 2, 3],
        "dropout": [0.0, 0.1, 0.2],
        "lr": [0.002, 0.003],
        "batch_size": [32, 64],
        "weight_decay": [0.0, 1e-4],
    },
    "LSTM_MLP": {
        "hidden_dim": [256],
        "num_layers": [1, 2, 3],
        "dropout": [0.0, 0.1, 0.2],
        "lr": [0.002, 0.003],
        "batch_size": [32, 64],
        "weight_decay": [0.0, 1e-4],
        "mlp_hidden_dim": [32, 64, 128],
        "mlp_num_layers": [1, 2],
    },
    "LSTM_KAN": {
        "hidden_dim": [256],
        "num_layers": [1, 2, 3],
        "dropout": [0.0, 0.1, 0.2],
        "lr": [0.002, 0.003],
        "batch_size": [32, 64],
        "weight_decay": [0.0, 1e-4],
        "kan_hidden_dim": [64],
        "grid_size": [3, 5, 7],
        "spline_order": [2, 3],
    },
}

MODEL_DISPLAY = {
    "LSTM_FC": "LSTM-FC",
    "LSTM_MLP": "LSTM-MLP",
    "LSTM_KAN": "LSTM-KAN",
}
MODEL_STEM = {
    "LSTM_FC": "lstm_fc",
    "LSTM_MLP": "lstm_mlp",
    "LSTM_KAN": "lstm_kan",
}

PARAM_COLUMNS = [
    "lookback",
    "hidden_dim",
    "num_layers",
    "dropout",
    "lr",
    "batch_size",
    "weight_decay",
    "mlp_hidden_dim",
    "mlp_num_layers",
    "kan_hidden_dim",
    "grid_size",
    "spline_order",
]
RESULT_COLUMNS = [
    "target",
    "model_name",
    "trial_id",
    *PARAM_COLUMNS,
    "best_epoch",
    "train_loss_at_best_epoch",
    "val_loss_at_best_epoch",
    "validation_nRMSE",
    "validation_nMAE",
    "validation_NSE",
    "validation_KGE",
]


class LSTMFC(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        hidden = self.dropout(out[:, -1, :])
        return self.fc(hidden)


class LSTMMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        mlp_hidden_dim: int,
        mlp_num_layers: int,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        layers = []
        in_dim = hidden_dim
        for _ in range(mlp_num_layers):
            layers.extend([nn.Linear(in_dim, mlp_hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = mlp_hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.mlp(out[:, -1, :])


class LSTMKAN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        kan_hidden_dim: int | None,
        grid_size: int,
        spline_order: int,
    ):
        super().__init__()
        if KAN is None:
            raise ImportError("未检测到 efficient-kan，请先安装 efficient-kan 后再运行 LSTM-KAN。")

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        layers_hidden = [hidden_dim, 1] if kan_hidden_dim is None else [hidden_dim, kan_hidden_dim, 1]
        self.kan = KAN(layers_hidden, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x):
        out, _ = self.lstm(x)
        hidden = self.dropout(out[:, -1, :])
        return self.kan(hidden)


def iter_grid(grid_space: dict) -> list[dict]:
    keys = list(grid_space.keys())
    return [dict(zip(keys, values)) for values in product(*(grid_space[k] for k in keys))]


def build_model(model_name: str, input_dim: int, params: dict) -> nn.Module:
    if model_name == "LSTM_FC":
        return LSTMFC(input_dim, params["hidden_dim"], params["num_layers"], params["dropout"])
    if model_name == "LSTM_MLP":
        return LSTMMLP(
            input_dim,
            params["hidden_dim"],
            params["num_layers"],
            params["dropout"],
            params["mlp_hidden_dim"],
            params["mlp_num_layers"],
        )
    if model_name == "LSTM_KAN":
        return LSTMKAN(
            input_dim,
            params["hidden_dim"],
            params["num_layers"],
            params["dropout"],
            params["kan_hidden_dim"],
            params["grid_size"],
            params["spline_order"],
        )
    raise ValueError(f"未知模型: {model_name}")


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


def params_text(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def output_paths(target: str, model_name: str) -> dict[str, Path]:
    stem = MODEL_STEM[model_name]
    return {
        "grid": OUTPUT_DIR / "grid_search" / target / f"{stem}_grid_results.csv",
        "loss_dir": OUTPUT_DIR / "loss_history" / target / model_name,
        "best_params": OUTPUT_DIR / "best_params" / target / f"best_params_{stem}.json",
        "best_model": OUTPUT_DIR / "best_models" / target / f"best_{stem}.pt",
        "pred_dir": OUTPUT_DIR / "predictions" / target,
    }


def grid_combination_count(model_name: str) -> int:
    count = 1
    for values in GRID_SPACES[model_name].values():
        count *= len(values)
    return count


def should_force_rerun_task(target: str, model_name: str) -> bool:
    return target in FORCE_RERUN_TARGETS and target not in PROTECTED_TARGETS


def csv_contains_task(path: Path, target: str, model_name: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        df = pd.read_csv(
            path,
            usecols=lambda col: col in {"target", "algae_type", "model_name", "dataset"},
        )
    except Exception:
        return True

    target_key = "target" if "target" in df.columns else "algae_type"
    if target_key not in df.columns or "model_name" not in df.columns:
        return False
    return bool(((df[target_key] == target) & (df["model_name"] == model_name)).any())


def remove_task_rows_from_csv(path: Path, target: str, model_name: str) -> None:
    if target in PROTECTED_TARGETS:
        raise ValueError(f"Refusing to modify protected target rows: {target}")
    if not path.exists() or path.stat().st_size == 0:
        return
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return

    target_key = "target" if "target" in df.columns else "algae_type"
    if target_key not in df.columns or "model_name" not in df.columns:
        return

    mask = (df[target_key] == target) & (df["model_name"] == model_name)
    if not mask.any():
        return
    df.loc[~mask].to_csv(path, index=False, encoding="utf-8-sig")


def reset_grid_outputs_for_rerun(target: str, model_name: str) -> None:
    if not should_force_rerun_task(target, model_name):
        return
    paths = output_paths(target, model_name)
    remove_task_rows_from_csv(paths["grid"], target, model_name)
    all_results_path = OUTPUT_DIR / "grid_search" / "all_targets_all_models_grid_results.csv"
    remove_task_rows_from_csv(all_results_path, target, model_name)


def reset_metric_outputs_for_rerun(target: str, model_name: str) -> None:
    if not should_force_rerun_task(target, model_name):
        return
    metrics_path = OUTPUT_DIR / "metrics" / "best_metrics_all_targets.csv"
    remove_task_rows_from_csv(metrics_path, target, model_name)


def existing_task_result_reason(target: str, model_name: str) -> str | None:
    paths = output_paths(target, model_name)
    if paths["best_params"].exists():
        return f"best params exists: {paths['best_params']}"
    if paths["best_model"].exists():
        return f"best model exists: {paths['best_model']}"
    if paths["grid"].exists() and paths["grid"].stat().st_size > 0:
        return f"grid result exists: {paths['grid']}"

    all_results_path = OUTPUT_DIR / "grid_search" / "all_targets_all_models_grid_results.csv"
    if csv_contains_task(all_results_path, target, model_name):
        return f"merged grid result exists: {all_results_path}"

    metrics_path = OUTPUT_DIR / "metrics" / "best_metrics_all_targets.csv"
    if csv_contains_task(metrics_path, target, model_name):
        return f"metrics result exists: {metrics_path}"

    stem = MODEL_STEM[model_name]
    for dataset in ["train", "val", "test"]:
        pred_path = paths["pred_dir"] / f"{stem}_best_predictions_{dataset}.csv"
        if pred_path.exists():
            return f"prediction result exists: {pred_path}"
    return None


def validate_run_config() -> None:
    for target in TARGETS_TO_RUN:
        if target not in TARGET_MAP:
            raise ValueError(f"Unknown target in TARGETS_TO_RUN: {target}")
    for model_name in MODELS_TO_RUN:
        if model_name not in GRID_SPACES:
            raise ValueError(f"Unknown model in MODELS_TO_RUN: {model_name}")


def build_run_tasks() -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    tasks_to_run = []
    skipped_tasks = []
    for target in TARGETS_TO_RUN:
        for model_name in MODELS_TO_RUN:
            if target in PROTECTED_TARGETS:
                skipped_tasks.append((target, model_name, "protected target"))
                continue
            force_rerun = should_force_rerun_task(target, model_name)
            reason = (
                existing_task_result_reason(target, model_name)
                if SKIP_EXISTING and not force_rerun
                else None
            )
            if reason is not None:
                skipped_tasks.append((target, model_name, reason))
            else:
                tasks_to_run.append((target, model_name))
    return tasks_to_run, skipped_tasks


def print_run_plan(
    tasks_to_run: list[tuple[str, str]],
    skipped_tasks: list[tuple[str, str, str]],
) -> None:
    grid_counts = {model_name: grid_combination_count(model_name) for model_name in MODELS_TO_RUN}
    total_combinations = sum(grid_counts[model_name] for _, model_name in tasks_to_run)

    print("\nGrid combinations per model:")
    for model_name in MODELS_TO_RUN:
        print(f"- {model_name}: {grid_counts[model_name]}")
    if FORCE_RERUN_TARGETS:
        force_targets = ", ".join(sorted(FORCE_RERUN_TARGETS))
        print(f"Force rerun targets: {force_targets}")
    if PROTECTED_TARGETS:
        protected_targets = ", ".join(sorted(PROTECTED_TARGETS))
        print(f"Protected targets: {protected_targets}")

    print(f"\nTasks to run ({len(tasks_to_run)}):")
    if tasks_to_run:
        for target, model_name in tasks_to_run:
            print(f"- {target} / {model_name}")
    else:
        print("- None")
    print(f"Total grid combinations to run: {total_combinations}")

    if skipped_tasks:
        print(f"\nSkipped existing tasks ({len(skipped_tasks)}):")
        for target, model_name, reason in skipped_tasks:
            print(f"- {target} / {model_name}: {reason}")


def ensure_output_dirs() -> None:
    for subdir in [
        "grid_search",
        "loss_history",
        "best_params",
        "best_models",
        "metrics",
        "predictions",
    ]:
        ensure_dir(OUTPUT_DIR / subdir)


def save_best_params(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_best_nrmse(path: Path) -> float:
    if not path.exists():
        return float("inf")
    info = json.loads(path.read_text(encoding="utf-8"))
    value = info.get("best_validation_nRMSE", float("inf"))
    return float(value) if value is not None else float("inf")


def evaluate_on_split(model, split_data: dict, y_scaler, batch_size: int, device: torch.device):
    pred_scaled = predict_scaled(model, split_data["X"], batch_size, device)
    true_value = inverse_log_target(split_data["y"], y_scaler)
    pred_value = inverse_log_target(pred_scaled, y_scaler)
    return true_value, pred_value, compute_metrics(true_value, pred_value)


def save_prediction_csv(path: Path, dates, target: str, true_value, pred_value) -> None:
    ensure_dir(path.parent)
    out = pd.DataFrame({"target": target, "true_value": true_value, "pred_value": pred_value})
    if dates is not None:
        out.insert(0, "date", pd.to_datetime(dates).strftime("%Y-%m-%d"))
    out.to_csv(path, index=False, encoding="utf-8-sig")


def upsert_metrics(metrics_path: Path, rows: list[dict]) -> None:
    ensure_dir(metrics_path.parent)
    new_df = pd.DataFrame(rows)
    if metrics_path.exists():
        old_df = pd.read_csv(
            metrics_path,
            usecols=lambda col: col in {"target", "algae_type", "model_name", "dataset"},
        )
        target_key = "target" if "target" in old_df.columns else "algae_type"
        if target_key in old_df.columns and {"model_name", "dataset"}.issubset(old_df.columns):
            existing_keys = set(zip(old_df[target_key], old_df["model_name"], old_df["dataset"]))
            new_df = new_df[
                ~new_df.apply(
                    lambda r: (r["target"], r["model_name"], r["dataset"]) in existing_keys,
                    axis=1,
                )
            ]
        if new_df.empty:
            return
        new_df.to_csv(metrics_path, mode="a", header=False, index=False, encoding="utf-8-sig")
        return
    new_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")


def run_grid_search_for_model(
    target: str,
    target_col: str,
    model_name: str,
    data: dict,
    input_dim: int,
    feature_cols: list[str],
    device: torch.device,
) -> None:
    paths = output_paths(target, model_name)
    grid = iter_grid(GRID_SPACES[model_name])
    all_results_path = OUTPUT_DIR / "grid_search" / "all_targets_all_models_grid_results.csv"
    force_rerun = should_force_rerun_task(target, model_name)
    if force_rerun:
        reset_grid_outputs_for_rerun(target, model_name)
        completed = set()
        best_nrmse = float("inf")
        print(f"[{target}][{MODEL_DISPLAY[model_name]}] Force rerun enabled; old non-protected rows are ignored.")
    else:
        completed = load_completed_trials(paths["grid"], target, model_name)
        best_nrmse = load_best_nrmse(paths["best_params"])

    for idx, search_params in enumerate(grid, start=1):
        trial_id = f"trial_{idx:04d}"
        if trial_id in completed:
            continue

        params = {"lookback": LOOKBACK, **search_params}
        set_seed(SEED)
        model = build_model(model_name, input_dim, params)
        train_loader = make_loader(
            data["train"]["X"], data["train"]["y"], params["batch_size"], True, seed=SEED
        )
        val_loader = make_loader(
            data["val"]["X"], data["val"]["y"], params["batch_size"], False, seed=SEED
        )

        history, best_state, best_epoch, train_loss_best, val_loss_best = train_one_model(
            model,
            train_loader,
            val_loader,
            params["lr"],
            MAX_EPOCHS,
            EARLY_STOPPING_PATIENCE,
            device,
            weight_decay=params.get("weight_decay", 0.0),
        )
        loss_path = paths["loss_dir"] / f"{trial_id}_loss.csv"
        save_loss_history(history, loss_path)

        val_true, val_pred, val_metrics = evaluate_on_split(
            model, data["val"], data["y_scaler"], params["batch_size"], device
        )

        row = {
            "target": target,
            "model_name": model_name,
            "trial_id": trial_id,
            **params,
            "best_epoch": best_epoch,
            "train_loss_at_best_epoch": train_loss_best,
            "val_loss_at_best_epoch": val_loss_best,
            "validation_nRMSE": val_metrics["nRMSE"],
            "validation_nMAE": val_metrics["nMAE"],
            "validation_NSE": val_metrics["NSE"],
            "validation_KGE": val_metrics["KGE"],
        }
        append_csv_row(paths["grid"], row, RESULT_COLUMNS)
        append_csv_row(all_results_path, row, RESULT_COLUMNS)

        updated = val_metrics["nRMSE"] < best_nrmse
        if updated:
            best_nrmse = val_metrics["nRMSE"]
            ensure_dir(paths["best_model"].parent)
            torch.save(best_state, paths["best_model"])
            save_best_params(
                paths["best_params"],
                {
                    "target": target,
                    "target_col": target_col,
                    "model_name": model_name,
                    "trial_id": trial_id,
                    "feature_cols": feature_cols,
                    "best_hyperparameters": params,
                    "best_validation_nRMSE": val_metrics["nRMSE"],
                    "best_validation_nMAE": val_metrics["nMAE"],
                    "best_validation_NSE": val_metrics["NSE"],
                    "best_validation_KGE": val_metrics["KGE"],
                    "best_epoch": best_epoch,
                },
            )

        mark = " [Best Updated]" if updated else ""
        print(
            f"[{target}][{MODEL_DISPLAY[model_name]}] Trial {idx}/{len(grid)} | "
            f"params={params_text(params)} | val nRMSE={metric_text(val_metrics['nRMSE'])} | "
            f"best so far={metric_text(best_nrmse)}{mark}"
        )


def evaluate_best_model(
    target: str,
    model_name: str,
    data: dict,
    input_dim: int,
    device: torch.device,
) -> None:
    paths = output_paths(target, model_name)
    if not paths["best_params"].exists() or not paths["best_model"].exists():
        print(f"[{target}][{MODEL_DISPLAY[model_name]}] 未找到最佳模型，跳过最终评价。")
        return

    best_info = json.loads(paths["best_params"].read_text(encoding="utf-8"))
    params = best_info["best_hyperparameters"]
    model = build_model(model_name, input_dim, params)
    state = torch.load(paths["best_model"], map_location="cpu")
    model.load_state_dict(state)

    metric_rows = []
    for dataset in ["train", "val", "test"]:
        true_value, pred_value, metrics = evaluate_on_split(
            model, data[dataset], data["y_scaler"], params["batch_size"], device
        )
        pred_path = paths["pred_dir"] / f"{MODEL_STEM[model_name]}_best_predictions_{dataset}.csv"
        save_prediction_csv(pred_path, data[dataset]["dates"], target, true_value, pred_value)
        metric_rows.append(
            {
                "target": target,
                "model_name": model_name,
                "dataset": dataset,
                **metrics,
            }
        )

    reset_metric_outputs_for_rerun(target, model_name)
    upsert_metrics(OUTPUT_DIR / "metrics" / "best_metrics_all_targets.csv", metric_rows)


def main() -> None:
    set_seed(SEED)
    ensure_output_dirs()
    validate_run_config()
    tasks_to_run, skipped_tasks = build_run_tasks()
    print_run_plan(tasks_to_run, skipped_tasks)

    if any(model_name == "LSTM_KAN" for _, model_name in tasks_to_run) and KAN is None:
        raise ImportError("efficient-kan is required before running LSTM-KAN.")
    if not tasks_to_run:
        print("\nNo new tasks to run. Existing grid-search outputs are preserved.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df, actual_data_dir, date_col = load_data_splits(
        DATA_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV
    )
    # 使用 common.data_utils.get_clean_feature_cols 构建去冗余特征列：
    # 有 log 版本的变量优先使用 log 列，避免原始变量和 log 变量重复输入。
    feature_cols = get_clean_feature_cols(train_df, date_col)
    print(f"Data: {actual_data_dir} | device={device} | features={len(feature_cols)}")

    for target in TARGETS_TO_RUN:
        model_names = [model_name for task_target, model_name in tasks_to_run if task_target == target]
        if not model_names:
            continue
        target_col = TARGET_MAP[target]
        print(f"\nCurrent target: {target} ({target_col})")
        data = prepare_single_target_data(
            train_df, val_df, test_df, feature_cols, target_col, LOOKBACK, HORIZON, date_col
        )
        input_dim = data["train"]["X"].shape[-1]

        for model_name in model_names:
            print(f"Current model: {MODEL_DISPLAY[model_name]}")
            run_grid_search_for_model(
                target, target_col, model_name, data, input_dim, feature_cols, device
            )
            evaluate_best_model(target, model_name, data, input_dim, device)


if __name__ == "__main__":
    main()
