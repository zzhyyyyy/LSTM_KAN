from __future__ import annotations

import copy
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
    infer_feature_cols,
    inverse_log_target,
    load_completed_trials,
    load_data_splits,
    prepare_single_target_data,
)
from common.metrics_utils import compute_metrics, rmse  # noqa: E402
from common.seed_utils import set_seed  # noqa: E402
from common.train_utils import make_loader, predict_scaled, save_loss_history  # noqa: E402

try:
    from efficient_kan import KAN
except ImportError:
    KAN = None


DATA_DIR = BASE_DIR / "data"
TRAIN_CSV = "train_model_input.csv"
VAL_CSV = "val_model_input.csv"
TEST_CSV = "test_model_input.csv"

OUTPUT_DIR = SCRIPT_DIR / "kan_full_grid_search_outputs"
MODEL_NAME = "KAN"
MODEL_STEM = "kan"

SEED = 42
LOOKBACK = 30
HORIZON = 1
MAX_EPOCHS = 200
PATIENCE = 20
SKIP_COMPLETED_TRIALS = True

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

# None uses common.data_utils.get_clean_feature_cols to avoid mixing raw and log scales.
FEATURE_COLS = None

GRID_SPACE = {
    "lookback": [LOOKBACK],
    "kan_hidden_dim": [32, 64, 128],
    "grid_size": [3, 5, 7],
    "spline_order": [2, 3],
    "dropout": [0.0, 0.1, 0.2],
    "lr": [0.001, 0.002, 0.003],
    "batch_size": [32, 64],
    "weight_decay": [0.0, 1e-4],
}

PARAM_COLUMNS = [
    "lookback",
    "kan_hidden_dim",
    "grid_size",
    "spline_order",
    "dropout",
    "lr",
    "batch_size",
    "weight_decay",
]

GRID_RESULT_COLUMNS = [
    "target",
    "model_name",
    "trial_id",
    *PARAM_COLUMNS,
    "best_epoch",
    "train_loss_at_best_epoch",
    "val_loss_at_best_epoch",
    "val_RMSE_at_best_epoch",
    "validation_RMSE",
    "validation_NSE",
    "validation_KGE",
    "validation_nRMSE",
    "validation_nMAE",
]

METRIC_COLUMNS = [
    "target",
    "model_name",
    "dataset",
    "RMSE",
    "NSE",
    "KGE",
    "nRMSE",
    "nMAE",
]

SUMMARY_PARAM_COLUMNS = [
    "target",
    "target_col",
    "model_name",
    "trial_id",
    *PARAM_COLUMNS,
    "best_epoch",
    "best_validation_RMSE",
    "best_validation_NSE",
    "best_validation_KGE",
    "best_validation_nRMSE",
    "best_validation_nMAE",
]


class KANRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        kan_hidden_dim: int = 64,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if KAN is None:
            raise ImportError("efficient-kan is required before running KAN grid search.")

        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        layers_hidden = [input_dim, kan_hidden_dim, output_dim]
        self.kan = KAN(layers_hidden, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        return self.kan(x)


def iter_grid(grid_space: dict) -> list[dict]:
    keys = list(grid_space.keys())
    return [dict(zip(keys, values)) for values in product(*(grid_space[k] for k in keys))]


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
    return ", ".join(f"{key}={value}" for key, value in params.items())


def _state_dict_to_cpu(state_dict: dict) -> dict:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def output_paths(target: str) -> dict[str, Path]:
    return {
        "grid": OUTPUT_DIR / "grid_search" / target / f"{MODEL_STEM}_grid_results.csv",
        "loss_dir": OUTPUT_DIR / "loss_history" / target / MODEL_NAME,
        "best_params": OUTPUT_DIR / "best_params" / target / f"best_params_{MODEL_STEM}.json",
        "best_model": OUTPUT_DIR / "best_models" / target / f"best_{MODEL_STEM}.pt",
        "pred_dir": OUTPUT_DIR / "predictions" / target,
    }


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


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def load_best_rmse(path: Path) -> float:
    if not path.exists():
        return float("inf")
    info = json.loads(path.read_text(encoding="utf-8"))
    value = info.get("best_validation_RMSE", float("inf"))
    return float(value) if value is not None else float("inf")


def validate_run_config() -> None:
    missing_targets = [target for target in TARGETS_TO_RUN if target not in TARGET_MAP]
    if missing_targets:
        raise ValueError(f"Unknown targets in TARGETS_TO_RUN: {missing_targets}")
    if GRID_SPACE.get("lookback") != [LOOKBACK]:
        raise ValueError("This script prepares data once per target, so GRID_SPACE['lookback'] must be [LOOKBACK].")


@torch.no_grad()
def evaluate_scaled_loss_and_rmse(
    model: nn.Module,
    split_data: dict,
    y_scaler,
    batch_size: int,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    loader = make_loader(split_data["X"], split_data["y"], batch_size, shuffle=False, seed=SEED)
    model.to(device)
    model.eval()

    loss_sum, n_obs = 0.0, 0
    preds, truths = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb_device = yb.to(device)
        output = model(xb)
        loss = criterion(output, yb_device)
        loss_sum += loss.item() * xb.size(0)
        n_obs += xb.size(0)
        preds.append(output.detach().cpu().numpy())
        truths.append(yb.numpy())

    val_loss = loss_sum / max(n_obs, 1)
    true_value = inverse_log_target(np.vstack(truths), y_scaler)
    pred_value = inverse_log_target(np.vstack(preds), y_scaler)
    return val_loss, rmse(true_value, pred_value)


def train_one_kan_model(
    model: nn.Module,
    train_loader,
    val_data: dict,
    y_scaler,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[list[dict], dict, int, float, float, float]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    best_state = None
    best_epoch = 0
    best_val_rmse = float("inf")
    train_loss_at_best = float("inf")
    val_loss_at_best = float("inf")
    wait = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_sum, train_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * xb.size(0)
            train_n += xb.size(0)

        train_loss = train_sum / max(train_n, 1)
        val_loss, val_rmse = evaluate_scaled_loss_and_rmse(
            model, val_data, y_scaler, batch_size, criterion, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_RMSE": val_rmse,
            }
        )

        improved = best_state is None or (
            np.isfinite(val_rmse) and val_rmse < best_val_rmse - 1e-12
        )
        if improved:
            best_state = _state_dict_to_cpu(model.state_dict())
            best_epoch = epoch
            best_val_rmse = val_rmse
            train_loss_at_best = train_loss
            val_loss_at_best = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        best_state = _state_dict_to_cpu(model.state_dict())
    model.load_state_dict(copy.deepcopy(best_state))
    return history, best_state, best_epoch, train_loss_at_best, val_loss_at_best, best_val_rmse


def build_model(input_dim: int, params: dict) -> nn.Module:
    return KANRegressor(
        input_dim=input_dim,
        output_dim=1,
        kan_hidden_dim=params["kan_hidden_dim"],
        grid_size=params["grid_size"],
        spline_order=params["spline_order"],
        dropout=params["dropout"],
    )


def evaluate_on_split(
    model: nn.Module,
    split_data: dict,
    y_scaler,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pred_scaled = predict_scaled(model, split_data["X"], batch_size, device)
    true_value = inverse_log_target(split_data["y"], y_scaler)
    pred_value = inverse_log_target(pred_scaled, y_scaler)
    metrics = {"RMSE": rmse(true_value, pred_value), **compute_metrics(true_value, pred_value)}
    return true_value, pred_value, metrics


def save_prediction_csv(path: Path, dates, target: str, true_value, pred_value) -> None:
    ensure_dir(path.parent)
    out = pd.DataFrame({"target": target, "true_value": true_value, "pred_value": pred_value})
    if dates is not None:
        out.insert(0, "date", pd.to_datetime(dates).strftime("%Y-%m-%d"))
    out.to_csv(path, index=False, encoding="utf-8-sig")


def run_grid_search_for_target(
    target: str,
    target_col: str,
    data: dict,
    input_dim: int,
    feature_cols: list[str],
    device: torch.device,
) -> None:
    paths = output_paths(target)
    grid = iter_grid(GRID_SPACE)
    all_results_path = OUTPUT_DIR / "grid_search" / "all_targets_kan_grid_results.csv"
    completed = (
        load_completed_trials(paths["grid"], target, MODEL_NAME)
        if SKIP_COMPLETED_TRIALS
        else set()
    )
    best_rmse = load_best_rmse(paths["best_params"])

    for idx, params in enumerate(grid, start=1):
        trial_id = f"trial_{idx:04d}"
        if trial_id in completed:
            continue

        set_seed(SEED)
        model = build_model(input_dim, params)
        train_loader = make_loader(
            data["train"]["X"],
            data["train"]["y"],
            params["batch_size"],
            shuffle=True,
            seed=SEED,
        )

        history, best_state, best_epoch, train_loss_best, val_loss_best, val_rmse_best = (
            train_one_kan_model(
                model=model,
                train_loader=train_loader,
                val_data=data["val"],
                y_scaler=data["y_scaler"],
                lr=params["lr"],
                weight_decay=params["weight_decay"],
                batch_size=params["batch_size"],
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
                device=device,
            )
        )

        loss_path = paths["loss_dir"] / f"{trial_id}_loss.csv"
        save_loss_history(history, loss_path)

        _, _, val_metrics = evaluate_on_split(
            model, data["val"], data["y_scaler"], params["batch_size"], device
        )
        row = {
            "target": target,
            "model_name": MODEL_NAME,
            "trial_id": trial_id,
            **params,
            "best_epoch": best_epoch,
            "train_loss_at_best_epoch": train_loss_best,
            "val_loss_at_best_epoch": val_loss_best,
            "val_RMSE_at_best_epoch": val_rmse_best,
            "validation_RMSE": val_metrics["RMSE"],
            "validation_NSE": val_metrics["NSE"],
            "validation_KGE": val_metrics["KGE"],
            "validation_nRMSE": val_metrics["nRMSE"],
            "validation_nMAE": val_metrics["nMAE"],
        }
        append_csv_row(paths["grid"], row, GRID_RESULT_COLUMNS)
        append_csv_row(all_results_path, row, GRID_RESULT_COLUMNS)

        updated = np.isfinite(val_metrics["RMSE"]) and val_metrics["RMSE"] < best_rmse
        if updated:
            best_rmse = val_metrics["RMSE"]
            ensure_dir(paths["best_model"].parent)
            torch.save(best_state, paths["best_model"])
            save_json(
                paths["best_params"],
                {
                    "target": target,
                    "target_col": target_col,
                    "model_name": MODEL_NAME,
                    "trial_id": trial_id,
                    "feature_cols": feature_cols,
                    "best_hyperparameters": params,
                    "best_epoch": best_epoch,
                    "best_validation_RMSE": val_metrics["RMSE"],
                    "best_validation_NSE": val_metrics["NSE"],
                    "best_validation_KGE": val_metrics["KGE"],
                    "best_validation_nRMSE": val_metrics["nRMSE"],
                    "best_validation_nMAE": val_metrics["nMAE"],
                },
            )

        mark = " [Best Updated]" if updated else ""
        print(
            f"Trial {idx}/{len(grid)} | params: {params_text(params)} | "
            f"val_RMSE: {metric_text(val_metrics['RMSE'])} | "
            f"best_val_RMSE: {metric_text(best_rmse)}{mark}"
        )


def evaluate_best_model(
    target: str,
    data: dict,
    input_dim: int,
    device: torch.device,
) -> tuple[list[dict], dict | None]:
    paths = output_paths(target)
    if not paths["best_params"].exists() or not paths["best_model"].exists():
        print(f"Finished target {target}. No best model found.")
        return [], None

    best_info = json.loads(paths["best_params"].read_text(encoding="utf-8"))
    params = best_info["best_hyperparameters"]
    model = build_model(input_dim, params)
    state = torch.load(paths["best_model"], map_location="cpu")
    model.load_state_dict(state)

    metric_rows = []
    for dataset in ["train", "val", "test"]:
        true_value, pred_value, metrics = evaluate_on_split(
            model, data[dataset], data["y_scaler"], params["batch_size"], device
        )
        pred_path = paths["pred_dir"] / f"{MODEL_STEM}_best_predictions_{dataset}.csv"
        save_prediction_csv(pred_path, data[dataset]["dates"], target, true_value, pred_value)
        metric_rows.append(
            {
                "target": target,
                "model_name": MODEL_NAME,
                "dataset": dataset,
                "RMSE": metrics["RMSE"],
                "NSE": metrics["NSE"],
                "KGE": metrics["KGE"],
                "nRMSE": metrics["nRMSE"],
                "nMAE": metrics["nMAE"],
            }
        )

    print(
        f"Finished target {target}. Best val RMSE = "
        f"{metric_text(best_info['best_validation_RMSE'])}"
    )
    return metric_rows, best_info


def best_params_summary_row(best_info: dict) -> dict:
    params = best_info["best_hyperparameters"]
    row = {
        "target": best_info["target"],
        "target_col": best_info["target_col"],
        "model_name": best_info["model_name"],
        "trial_id": best_info["trial_id"],
        "best_epoch": best_info["best_epoch"],
        "best_validation_RMSE": best_info["best_validation_RMSE"],
        "best_validation_NSE": best_info["best_validation_NSE"],
        "best_validation_KGE": best_info["best_validation_KGE"],
        "best_validation_nRMSE": best_info["best_validation_nRMSE"],
        "best_validation_nMAE": best_info["best_validation_nMAE"],
    }
    row.update({column: params.get(column, np.nan) for column in PARAM_COLUMNS})
    return row


def save_summary_files(metric_rows: list[dict], best_infos: list[dict]) -> None:
    metrics_df = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS)
    metrics_path = OUTPUT_DIR / "metrics" / "best_metrics_all_targets.csv"
    ensure_dir(metrics_path.parent)
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(OUTPUT_DIR / "summary_best_metrics.csv", index=False, encoding="utf-8-sig")

    params_df = pd.DataFrame(
        [best_params_summary_row(info) for info in best_infos],
        columns=SUMMARY_PARAM_COLUMNS,
    )
    params_df.to_csv(OUTPUT_DIR / "summary_best_params.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    validate_run_config()
    if KAN is None:
        raise ImportError("efficient-kan is required before running KAN grid search.")

    set_seed(SEED)
    ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = iter_grid(GRID_SPACE)
    total_trials = len(grid) * len(TARGETS_TO_RUN)

    train_df, val_df, test_df, actual_data_dir, date_col = load_data_splits(
        DATA_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV
    )
    feature_cols = infer_feature_cols(train_df, FEATURE_COLS, date_col)

    print(f"Data: {actual_data_dir} | device={device} | features={len(feature_cols)}")
    print(f"Grid combinations per target: {len(grid)}")
    print(f"Total planned trials: {total_trials}")

    all_metric_rows = []
    best_infos = []
    for target_idx, target in enumerate(TARGETS_TO_RUN, start=1):
        target_col = TARGET_MAP[target]
        print(f"\n[{target_idx}/{len(TARGETS_TO_RUN)}] Target: {target}")
        data = prepare_single_target_data(
            train_df,
            val_df,
            test_df,
            feature_cols,
            target_col,
            LOOKBACK,
            HORIZON,
            date_col,
        )
        input_dim = data["train"]["X"].shape[1] * data["train"]["X"].shape[2]

        run_grid_search_for_target(target, target_col, data, input_dim, feature_cols, device)
        metric_rows, best_info = evaluate_best_model(target, data, input_dim, device)
        all_metric_rows.extend(metric_rows)
        if best_info is not None:
            best_infos.append(best_info)

    save_summary_files(all_metric_rows, best_infos)
    print(f"\nSaved summary files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
