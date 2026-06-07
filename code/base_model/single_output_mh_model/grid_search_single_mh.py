"""
单输出·多步基础模型分步搜索（LSTM-FC / LSTM-MLP / LSTM-KAN）。

每个目标各训练一个模型：输入 13 个原始特征(不含 Algae_Sum)，
一次输出该目标的 t+1/t+2/t+3（output_dim = len(HORIZONS) = 3）。
4 架构 × 5 目标 = 20 个模型。设定与 multi_output_model 完全一致：
原始尺度、滚动多步、坐标下降分步优化、7:1:2、选优用跨 horizon 平均 val nRMSE。

复用 multi_output_model 的超参搜索空间(BASELINES/PARAM_ORDER)与公共数据/训练工具。
烟雾测试：MULTI_SMOKE=1 → 每超参只留基线 + 2 epoch。
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
    inverse_log_targets,
    inverse_targets,
    load_data_splits,
    prepare_multi_horizon_data,
)
from common.metrics_utils import compute_metrics  # noqa: E402
from common.seed_utils import set_seed  # noqa: E402
from common.train_utils import make_loader, predict_scaled_multi, save_loss_history, train_one_model  # noqa: E402
from models import HORIZONS, KAN, MODEL_DISPLAY, MODEL_STEM, TARGET_COLS_ORDER, TARGET_MAP_TRAIN, USE_LOG, build_model  # noqa: E402
from grid_search_multi_output import BASELINES, PARAM_ORDER, PARAM_COLUMNS, FEATURE_COLS_FN  # noqa: E402

# USE_LOG 控制目标 log1p+expm1；FEATURE_COLS_FN（由 models.LOG_INPUTS 决定）控制输入是否 log。
INVERSE_FN = inverse_log_targets if USE_LOG else inverse_targets


DATA_DIR = BASE_DIR / "data"
TRAIN_CSV, VAL_CSV, TEST_CSV = "train_model_input.csv", "val_model_input.csv", "test_model_input.csv"
OUTPUT_DIR = SCRIPT_DIR / "grid_search_outputs_single_mh"
SEED = 42
LOOKBACK = 30
MAX_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 20

SMOKE = bool(os.environ.get("MULTI_SMOKE"))
if SMOKE:
    MAX_EPOCHS = 2
    EARLY_STOPPING_PATIENCE = 2

TARGETS = list(TARGET_COLS_ORDER)
MODELS_TO_RUN = ["LSTM_FC", "LSTM_MLP", "LSTM_KAN"]

PER_HORIZON_NRMSE_COLS = [f"val_nRMSE_h{h}" for h in HORIZONS]
RESULT_COLUMNS = [
    "target", "model_name", "trial_id", "tuned_param", *PARAM_COLUMNS,
    "best_epoch", "train_loss_at_best_epoch", "val_loss_at_best_epoch",
    "validation_mean_nRMSE", "validation_mean_nMAE", "validation_mean_NSE", "validation_mean_KGE",
    *PER_HORIZON_NRMSE_COLS,
]
METRIC_COLUMNS = ["target", "model_name", "horizon", "dataset", "nRMSE", "nMAE", "NSE", "KGE"]


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if np.isfinite(v) else None
    return obj


def metric_text(v):
    return "nan" if v is None or not np.isfinite(v) else f"{v:.6f}"


def output_paths(target, model_name):
    stem = MODEL_STEM[model_name]
    return {
        "search": OUTPUT_DIR / "search" / target / f"{stem}_search_results.csv",
        "loss_dir": OUTPUT_DIR / "loss_history" / target / model_name,
        "best_params": OUTPUT_DIR / "best_params" / target / f"best_params_{stem}.json",
        "best_model": OUTPUT_DIR / "best_models" / target / f"best_{stem}.pt",
        "pred_dir": OUTPUT_DIR / "predictions" / target,
    }


def ensure_output_dirs():
    for sub in ["search", "loss_history", "best_params", "best_models", "metrics", "predictions"]:
        ensure_dir(OUTPUT_DIR / sub)


def evaluate_split(model, split, y_scaler, batch_size, device, output_dim):
    """单目标多步：pred (N,H)。逐 horizon 反标准化(单列)算指标。"""
    pred = predict_scaled_multi(model, split["X"], batch_size, device, output_dim)  # (N,H)
    true = split["y"]  # (N,H)
    per, pred_raw, true_raw = {}, np.zeros_like(pred), np.zeros_like(true)
    for h in range(len(HORIZONS)):
        p = INVERSE_FN(pred[:, [h]], y_scaler)[:, 0]
        t = INVERSE_FN(true[:, [h]], y_scaler)[:, 0]
        pred_raw[:, h], true_raw[:, h] = p, t
        per[h] = compute_metrics(t, p)
    return per, pred_raw, true_raw


def agg(per, key):
    return float(np.nanmean([per[h][key] for h in range(len(HORIZONS))]))


def save_predictions_long(path, dates, pred_raw, true_raw, target):
    ensure_dir(path.parent)
    frames = []
    for h_idx, horizon in enumerate(HORIZONS):
        date_h = pd.to_datetime(dates[:, h_idx]).strftime("%Y-%m-%d") if dates is not None else None
        frame = pd.DataFrame({"horizon": horizon, "target": target,
                              "true_value": true_raw[:, h_idx], "pred_value": pred_raw[:, h_idx]})
        if date_h is not None:
            frame.insert(0, "date", date_h)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")


def coordinate_descent(target, model_name, data, output_dim, device):
    paths = output_paths(target, model_name)
    baseline = BASELINES[model_name]
    param_order = PARAM_ORDER[model_name]
    if SMOKE:
        param_order = [(p, [baseline[p]]) for p, _ in param_order]
    input_dim = data["train"]["X"].shape[-1]
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
        model = build_model(model_name, input_dim, params, output_dim)
        tl = make_loader(data["train"]["X"], data["train"]["y"], params["batch_size"], True, seed=SEED)
        vl = make_loader(data["val"]["X"], data["val"]["y"], params["batch_size"], False, seed=SEED)
        history, best_state, best_epoch, train_loss, val_loss = train_one_model(
            model, tl, vl, params["lr"], MAX_EPOCHS, EARLY_STOPPING_PATIENCE, device,
            weight_decay=params.get("weight_decay", 0.0))
        save_loss_history(history, paths["loss_dir"] / f"{trial_id}_loss.csv")
        per, _, _ = evaluate_split(model, data["val"], data["y_scaler"], params["batch_size"], device, output_dim)
        mean_nrmse = agg(per, "nRMSE")
        row = {"target": target, "model_name": model_name, "trial_id": trial_id, "tuned_param": tuned,
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

    current = {"lookback": LOOKBACK, **baseline}
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


def run_target_model(target, model_name, data, output_dim, device):
    paths = output_paths(target, model_name)
    final = coordinate_descent(target, model_name, data, output_dim, device)
    params = final["params"]
    ensure_dir(paths["best_model"].parent)
    torch.save(final["best_state"], paths["best_model"])
    per = final["per"]
    payload = {
        "target": target, "target_col": TARGET_MAP_TRAIN[target], "model_name": model_name,
        "model_display": MODEL_DISPLAY[model_name], "feature_cols": data["feature_cols"],
        "horizons": HORIZONS, "best_hyperparameters": params, "best_epoch": final["best_epoch"],
        "best_validation_val_loss": final["val_loss"], "best_validation_mean_nRMSE": final["mean_nrmse"],
        "best_validation_per_horizon_nRMSE": {f"h{HORIZONS[h]}": per[h]["nRMSE"] for h in range(len(HORIZONS))},
    }
    ensure_dir(paths["best_params"].parent)
    paths["best_params"].write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    set_seed(SEED)
    model = build_model(model_name, data["train"]["X"].shape[-1], params, output_dim)
    model.load_state_dict(torch.load(paths["best_model"], map_location="cpu"))
    rows = []
    for dataset in ["train", "val", "test"]:
        per, pred_raw, true_raw = evaluate_split(model, data[dataset], data["y_scaler"], params["batch_size"], device, output_dim)
        save_predictions_long(paths["pred_dir"] / f"{MODEL_STEM[model_name]}_best_predictions_{dataset}.csv",
                              data[dataset]["dates"], pred_raw, true_raw, target)
        for h in range(len(HORIZONS)):
            rows.append({"target": target, "model_name": model_name, "horizon": HORIZONS[h], "dataset": dataset, **per[h]})
    return rows


def main():
    set_seed(SEED)
    ensure_output_dirs()
    if "LSTM_KAN" in MODELS_TO_RUN and KAN is None:
        raise ImportError("efficient-kan is required before running LSTM-KAN.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df, actual_data_dir, date_col = load_data_splits(DATA_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV)
    feature_cols = FEATURE_COLS_FN(train_df, date_col)
    output_dim = len(HORIZONS)
    print(f"Data: {actual_data_dir} | device={device} | raw features={len(feature_cols)} | "
          f"targets={len(TARGETS)} | horizons={HORIZONS} | output_dim/model={output_dim} | smoke={SMOKE}", flush=True)

    all_rows = []
    for target in TARGETS:
        data = prepare_multi_horizon_data(train_df, val_df, test_df, feature_cols, [TARGET_MAP_TRAIN[target]], LOOKBACK, HORIZONS, date_col)
        data["feature_cols"] = feature_cols
        for model_name in MODELS_TO_RUN:
            print(f"\n=== {target} / {MODEL_DISPLAY[model_name]} (staged) ===", flush=True)
            rows = run_target_model(target, model_name, data, output_dim, device)
            all_rows.extend(rows)
            best = json.loads(output_paths(target, model_name)["best_params"].read_text(encoding="utf-8"))
            print(f"  best val mean nRMSE={metric_text(best['best_validation_mean_nRMSE'])}", flush=True)

    if all_rows:
        mp = OUTPUT_DIR / "metrics" / "best_metrics_all.csv"
        pd.DataFrame(all_rows, columns=METRIC_COLUMNS).to_csv(mp, index=False, encoding="utf-8-sig")
        print(f"\nSaved best metrics: {mp}", flush=True)
    print(f"Output directory: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
