"""
单输出·多步 纯 KAN 分步搜索（第 4 个架构）。

每个目标一个纯 KAN：输入展平 (lookback*13)，一次输出该目标的 t+1/t+2/t+3。
设定与 LSTM 家族单输出版一致；复用 grid_search_single_mh 的评估/落盘工具
与 kan_full_grid_search_multi 的 KAN 超参搜索空间。
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
MULTI_DIR = BASE_DIR / "multi_output_model"
for _p in (BASE_DIR, MULTI_DIR, SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.data_utils import (  # noqa: E402
    append_csv_row,
    ensure_dir,
    get_raw_feature_cols,
    load_data_splits,
    prepare_multi_horizon_data,
)
from common.seed_utils import set_seed  # noqa: E402
from common.train_utils import make_loader, save_loss_history, train_one_model  # noqa: E402
from models import HORIZONS, KAN, MODEL_DISPLAY, MODEL_STEM, TARGET_COLS_ORDER, TARGET_MAP_TRAIN, build_model  # noqa: E402
from kan_full_grid_search_multi import BASELINE as KAN_BASELINE, PARAM_ORDER as KAN_PARAM_ORDER  # noqa: E402
from grid_search_single_mh import (  # noqa: E402
    PARAM_COLUMNS,
    agg,
    evaluate_split,
    json_safe,
    metric_text,
    save_predictions_long,
)


DATA_DIR = BASE_DIR / "data"
TRAIN_CSV, VAL_CSV, TEST_CSV = "train_model_input.csv", "val_model_input.csv", "test_model_input.csv"
OUTPUT_DIR = SCRIPT_DIR / "kan_full_grid_search_single_mh_outputs"
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

TARGETS = list(TARGET_COLS_ORDER)
PER_HORIZON_NRMSE_COLS = [f"val_nRMSE_h{h}" for h in HORIZONS]
RESULT_COLUMNS = [
    "target", "model_name", "trial_id", "tuned_param", *PARAM_COLUMNS,
    "best_epoch", "train_loss_at_best_epoch", "val_loss_at_best_epoch",
    "validation_mean_nRMSE", "validation_mean_nMAE", "validation_mean_NSE", "validation_mean_KGE",
    *PER_HORIZON_NRMSE_COLS,
]
METRIC_COLUMNS = ["target", "model_name", "horizon", "dataset", "nRMSE", "nMAE", "NSE", "KGE"]


def output_paths(target):
    return {
        "search": OUTPUT_DIR / "search" / target / f"{STEM}_search_results.csv",
        "loss_dir": OUTPUT_DIR / "loss_history" / target / MODEL_NAME,
        "best_params": OUTPUT_DIR / "best_params" / target / f"best_params_{STEM}.json",
        "best_model": OUTPUT_DIR / "best_models" / target / f"best_{STEM}.pt",
        "pred_dir": OUTPUT_DIR / "predictions" / target,
    }


def ensure_output_dirs():
    for sub in ["search", "loss_history", "best_params", "best_models", "metrics", "predictions"]:
        ensure_dir(OUTPUT_DIR / sub)


def coordinate_descent(target, data, input_dim, output_dim, device):
    paths = output_paths(target)
    param_order = [(p, [KAN_BASELINE[p]]) for p, _ in KAN_PARAM_ORDER] if SMOKE else KAN_PARAM_ORDER
    cache, counter = {}, {"n": 0}

    def key(p):
        return tuple(sorted(p.items()))

    def evaluate(params, tuned):
        k = key(params)
        if k in cache:
            return cache[k]
        counter["n"] += 1
        trial_id = f"trial_{counter['n']:04d}"
        set_seed(SEED)
        model = build_model(MODEL_NAME, input_dim, params, output_dim)
        tl = make_loader(data["train"]["X"], data["train"]["y"], params["batch_size"], True, seed=SEED)
        vl = make_loader(data["val"]["X"], data["val"]["y"], params["batch_size"], False, seed=SEED)
        history, best_state, best_epoch, train_loss, val_loss = train_one_model(
            model, tl, vl, params["lr"], MAX_EPOCHS, PATIENCE, device, weight_decay=params.get("weight_decay", 0.0))
        save_loss_history(history, paths["loss_dir"] / f"{trial_id}_loss.csv")
        per, _, _ = evaluate_split(model, data["val"], data["y_scaler"], params["batch_size"], device, output_dim)
        mean_nrmse = agg(per, "nRMSE")
        row = {"target": target, "model_name": MODEL_NAME, "trial_id": trial_id, "tuned_param": tuned,
               **{c: params.get(c) for c in PARAM_COLUMNS},
               "best_epoch": best_epoch, "train_loss_at_best_epoch": train_loss, "val_loss_at_best_epoch": val_loss,
               "validation_mean_nRMSE": mean_nrmse, "validation_mean_nMAE": agg(per, "nMAE"),
               "validation_mean_NSE": agg(per, "NSE"), "validation_mean_KGE": agg(per, "KGE"),
               **{f"val_nRMSE_h{HORIZONS[h]}": per[h]["nRMSE"] for h in range(len(HORIZONS))}}
        append_csv_row(paths["search"], row, RESULT_COLUMNS)
        res = {"mean_nrmse": mean_nrmse, "val_loss": val_loss, "best_state": best_state,
               "best_epoch": best_epoch, "params": dict(params), "per": per}
        cache[k] = res
        return res

    current = dict(KAN_BASELINE)
    best = evaluate(current, "baseline")
    for pname, candidates in param_order:
        sbv, sbest = current[pname], cache[key(current)]
        for v in candidates:
            r = evaluate({**current, pname: v}, pname)
            if r["mean_nrmse"] < sbest["mean_nrmse"] - 1e-9 or (
                abs(r["mean_nrmse"] - sbest["mean_nrmse"]) <= 1e-9 and r["val_loss"] < sbest["val_loss"]):
                sbv, sbest = v, r
        current[pname] = sbv
        if sbest["mean_nrmse"] < best["mean_nrmse"]:
            best = sbest
    return cache[key(current)]


def main():
    if KAN is None:
        raise ImportError("efficient-kan is required before running KAN.")
    set_seed(SEED)
    ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df, actual_data_dir, date_col = load_data_splits(DATA_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV)
    feature_cols = get_raw_feature_cols(train_df, date_col)
    output_dim = len(HORIZONS)
    print(f"Data: {actual_data_dir} | device={device} | raw features={len(feature_cols)} | "
          f"targets={len(TARGETS)} | horizons={HORIZONS} | smoke={SMOKE}", flush=True)

    all_rows = []
    for target in TARGETS:
        data = prepare_multi_horizon_data(train_df, val_df, test_df, feature_cols, [TARGET_MAP_TRAIN[target]], LOOKBACK, HORIZONS, date_col)
        input_dim = data["train"]["X"].shape[1] * data["train"]["X"].shape[2]
        print(f"\n=== {target} / KAN (staged) === input_dim={input_dim}", flush=True)
        final = coordinate_descent(target, data, input_dim, output_dim, device)
        params = final["params"]
        paths = output_paths(target)
        ensure_dir(paths["best_model"].parent)
        torch.save(final["best_state"], paths["best_model"])
        per = final["per"]
        payload = {"target": target, "target_col": TARGET_MAP_TRAIN[target], "model_name": MODEL_NAME,
                   "model_display": MODEL_DISPLAY[MODEL_NAME], "feature_cols": feature_cols, "horizons": HORIZONS,
                   "best_hyperparameters": params, "best_epoch": final["best_epoch"],
                   "best_validation_val_loss": final["val_loss"], "best_validation_mean_nRMSE": final["mean_nrmse"],
                   "best_validation_per_horizon_nRMSE": {f"h{HORIZONS[h]}": per[h]["nRMSE"] for h in range(len(HORIZONS))}}
        ensure_dir(paths["best_params"].parent)
        paths["best_params"].write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  best val mean nRMSE={metric_text(final['mean_nrmse'])}", flush=True)

        set_seed(SEED)
        model = build_model(MODEL_NAME, input_dim, params, output_dim)
        model.load_state_dict(torch.load(paths["best_model"], map_location="cpu"))
        for dataset in ["train", "val", "test"]:
            per_d, pred_raw, true_raw = evaluate_split(model, data[dataset], data["y_scaler"], params["batch_size"], device, output_dim)
            save_predictions_long(paths["pred_dir"] / f"{STEM}_best_predictions_{dataset}.csv",
                                  data[dataset]["dates"], pred_raw, true_raw, target)
            for h in range(len(HORIZONS)):
                all_rows.append({"target": target, "model_name": MODEL_NAME, "horizon": HORIZONS[h], "dataset": dataset, **per_d[h]})

    pd.DataFrame(all_rows, columns=METRIC_COLUMNS).to_csv(OUTPUT_DIR / "metrics" / "best_metrics_kan.csv", index=False, encoding="utf-8-sig")
    print(f"Output directory: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
