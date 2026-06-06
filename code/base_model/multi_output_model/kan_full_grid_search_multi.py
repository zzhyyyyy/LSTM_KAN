"""
纯 KAN 多输出 + 多步 + 原始尺度，分步(坐标下降)超参数优化（第 4 个基础模型）。

与 LSTM 家族口径一致（原始 13 特征、目标 5 原始列、t+1/t+2/t+3 共 15 输出、
inverse_targets 只反标准化、选优用跨 (h,target) 平均 val nRMSE、坐标下降分步搜索）。
区别：输入展平，input_dim = lookback * n_features。独立输出目录（纯 KAN 仅作基础模型对比）。
"""

from __future__ import annotations

import json
import os
import sys
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
    get_raw_feature_cols,
    load_data_splits,
    prepare_multi_horizon_data,
)
from common.seed_utils import set_seed  # noqa: E402
from common.train_utils import make_loader, save_loss_history, train_one_model  # noqa: E402
from models import HORIZONS, KAN, MODEL_DISPLAY, MODEL_STEM, TARGET_COLS, TARGET_COLS_ORDER, build_model  # noqa: E402
from grid_search_multi_output import (  # noqa: E402
    METRIC_COLUMNS,
    PARAM_COLUMNS,
    agg_mean,
    evaluate_split,
    json_safe,
    metric_text,
    per_horizon_mean_nrmse,
    per_target_mean_nrmse,
    save_predictions_long,
)


DATA_DIR = BASE_DIR / "data"
TRAIN_CSV, VAL_CSV, TEST_CSV = "train_model_input.csv", "val_model_input.csv", "test_model_input.csv"
OUTPUT_DIR = SCRIPT_DIR / "kan_full_grid_search_multi_outputs"
MODEL_NAME = "KAN"
STEM = MODEL_STEM[MODEL_NAME]
SEED = 42
LOOKBACK = 30
MAX_EPOCHS = 200
PATIENCE = 20

SMOKE = bool(os.environ.get("MULTI_SMOKE"))
if SMOKE:
    MAX_EPOCHS = 2
    PATIENCE = 2

BASELINE = {"lookback": LOOKBACK, "kan_hidden_dim": 64, "grid_size": 5, "spline_order": 3,
            "lr": 0.002, "dropout": 0.0, "batch_size": 64, "weight_decay": 0.0}
PARAM_ORDER = [
    ("kan_hidden_dim", [32, 64, 128]),
    ("grid_size", [3, 5, 7]),
    ("spline_order", [2, 3]),
    ("lr", [0.001, 0.002, 0.003]),
    ("dropout", [0.0, 0.1, 0.2]),
    ("batch_size", [32, 64]),
    ("weight_decay", [0.0, 1e-4]),
]

PER_HORIZON_NRMSE_COLS = [f"val_nRMSE_h{h}" for h in HORIZONS]
PER_TARGET_NRMSE_COLS = [f"val_nRMSE_{name}" for name in TARGET_COLS_ORDER]
RESULT_COLUMNS = [
    "model_name", "trial_id", "tuned_param", *PARAM_COLUMNS,
    "best_epoch", "train_loss_at_best_epoch", "val_loss_at_best_epoch",
    "validation_mean_nRMSE", "validation_mean_nMAE", "validation_mean_NSE", "validation_mean_KGE",
    *PER_HORIZON_NRMSE_COLS, *PER_TARGET_NRMSE_COLS,
]


def output_paths() -> dict[str, Path]:
    return {
        "search": OUTPUT_DIR / "search" / f"{STEM}_search_results.csv",
        "loss_dir": OUTPUT_DIR / "loss_history" / MODEL_NAME,
        "best_params": OUTPUT_DIR / "best_params" / f"best_params_{STEM}.json",
        "best_model": OUTPUT_DIR / "best_models" / f"best_{STEM}.pt",
        "pred_dir": OUTPUT_DIR / "predictions" / MODEL_NAME,
    }


def ensure_output_dirs() -> None:
    for subdir in ["search", "loss_history", "best_params", "best_models", "metrics", "predictions"]:
        ensure_dir(OUTPUT_DIR / subdir)


def coordinate_descent(data, input_dim, output_dim, device):
    paths = output_paths()
    param_order = [(p, [BASELINE[p]]) for p, _ in PARAM_ORDER] if SMOKE else PARAM_ORDER
    cache: dict = {}
    counter = {"n": 0}

    def key(params):
        return tuple(sorted(params.items()))

    def evaluate(params, tuned_param):
        k = key(params)
        if k in cache:
            return cache[k]
        counter["n"] += 1
        trial_id = f"trial_{counter['n']:04d}"
        set_seed(SEED)
        model = build_model(MODEL_NAME, input_dim, params, output_dim)
        train_loader = make_loader(data["train"]["X"], data["train"]["y"], params["batch_size"], True, seed=SEED)
        val_loader = make_loader(data["val"]["X"], data["val"]["y"], params["batch_size"], False, seed=SEED)
        history, best_state, best_epoch, train_loss, val_loss = train_one_model(
            model, train_loader, val_loader, params["lr"], MAX_EPOCHS, PATIENCE, device,
            weight_decay=params.get("weight_decay", 0.0),
        )
        save_loss_history(history, paths["loss_dir"] / f"{trial_id}_loss.csv")
        per, _, _ = evaluate_split(model, data["val"], data["y_scaler"], params["batch_size"], device, output_dim)
        mean_nrmse = agg_mean(per, "nRMSE")
        ph, pt = per_horizon_mean_nrmse(per), per_target_mean_nrmse(per)
        row = {
            "model_name": MODEL_NAME, "trial_id": trial_id, "tuned_param": tuned_param,
            **{c: params.get(c) for c in PARAM_COLUMNS},
            "best_epoch": best_epoch, "train_loss_at_best_epoch": train_loss, "val_loss_at_best_epoch": val_loss,
            "validation_mean_nRMSE": mean_nrmse, "validation_mean_nMAE": agg_mean(per, "nMAE"),
            "validation_mean_NSE": agg_mean(per, "NSE"), "validation_mean_KGE": agg_mean(per, "KGE"),
            **{f"val_nRMSE_h{HORIZONS[i]}": ph[i] for i in range(len(HORIZONS))},
            **{f"val_nRMSE_{TARGET_COLS_ORDER[i]}": pt[i] for i in range(len(TARGET_COLS_ORDER))},
        }
        append_csv_row(paths["search"], row, RESULT_COLUMNS)
        result = {"mean_nrmse": mean_nrmse, "val_loss": val_loss, "best_state": best_state,
                  "best_epoch": best_epoch, "params": dict(params), "per": per}
        cache[k] = result
        return result

    current = dict(BASELINE)
    base_res = evaluate(current, "baseline")
    print(f"[KAN] baseline val mean nRMSE={metric_text(base_res['mean_nrmse'])}", flush=True)
    for pname, candidates in param_order:
        stage_best_value, stage_best = current[pname], cache[key(current)]
        for v in candidates:
            res = evaluate({**current, pname: v}, pname)
            if res["mean_nrmse"] < stage_best["mean_nrmse"] - 1e-9 or (
                abs(res["mean_nrmse"] - stage_best["mean_nrmse"]) <= 1e-9 and res["val_loss"] < stage_best["val_loss"]
            ):
                stage_best_value, stage_best = v, res
        current[pname] = stage_best_value
        print(f"[KAN] tuned {pname} -> {stage_best_value} | val mean nRMSE={metric_text(stage_best['mean_nrmse'])}", flush=True)
    return cache[key(current)]


def main() -> None:
    if KAN is None:
        raise ImportError("efficient-kan is required before running KAN.")
    set_seed(SEED)
    ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df, actual_data_dir, date_col = load_data_splits(DATA_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV)
    feature_cols = get_raw_feature_cols(train_df, date_col)
    data = prepare_multi_horizon_data(train_df, val_df, test_df, feature_cols, TARGET_COLS, LOOKBACK, HORIZONS, date_col)
    output_dim = data["train"]["y"].shape[1]
    input_dim = data["train"]["X"].shape[1] * data["train"]["X"].shape[2]  # 展平
    print(f"Data: {actual_data_dir} | device={device} | raw features={len(feature_cols)} | "
          f"horizons={HORIZONS} | output_dim={output_dim} | input_dim={input_dim} | smoke={SMOKE}", flush=True)

    final = coordinate_descent(data, input_dim, output_dim, device)
    params = final["params"]
    paths = output_paths()
    ensure_dir(paths["best_model"].parent)
    torch.save(final["best_state"], paths["best_model"])
    val_per = final["per"]
    payload = {
        "model_name": MODEL_NAME, "model_display": MODEL_DISPLAY[MODEL_NAME],
        "target_names": TARGET_COLS_ORDER, "target_cols": TARGET_COLS,
        "feature_cols": feature_cols, "horizons": HORIZONS,
        "best_hyperparameters": params, "best_epoch": final["best_epoch"],
        "best_validation_val_loss": final["val_loss"], "best_validation_mean_nRMSE": final["mean_nrmse"],
        "best_validation_mean_nMAE": agg_mean(val_per, "nMAE"),
        "best_validation_mean_NSE": agg_mean(val_per, "NSE"),
        "best_validation_mean_KGE": agg_mean(val_per, "KGE"),
        "best_validation_per_horizon_nRMSE": {f"h{HORIZONS[i]}": per_horizon_mean_nrmse(val_per)[i] for i in range(len(HORIZONS))},
        "best_validation_per_target_nRMSE": {TARGET_COLS_ORDER[i]: per_target_mean_nrmse(val_per)[i] for i in range(len(TARGET_COLS_ORDER))},
    }
    ensure_dir(paths["best_params"].parent)
    paths["best_params"].write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    set_seed(SEED)
    model = build_model(MODEL_NAME, input_dim, params, output_dim)
    model.load_state_dict(torch.load(paths["best_model"], map_location="cpu"))
    metric_rows = []
    for dataset in ["train", "val", "test"]:
        per, pred_raw, true_raw = evaluate_split(model, data[dataset], data["y_scaler"], params["batch_size"], device, output_dim)
        save_predictions_long(paths["pred_dir"] / f"{STEM}_best_predictions_{dataset}.csv",
                              data[dataset]["dates"], pred_raw, true_raw)
        for (h_idx, name), m in per.items():
            metric_rows.append({"model_name": MODEL_NAME, "horizon": HORIZONS[h_idx], "target": name, "dataset": dataset, **m})
    pd.DataFrame(metric_rows, columns=METRIC_COLUMNS).to_csv(OUTPUT_DIR / "metrics" / "best_metrics_kan.csv", index=False, encoding="utf-8-sig")
    print(f"Output directory: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
