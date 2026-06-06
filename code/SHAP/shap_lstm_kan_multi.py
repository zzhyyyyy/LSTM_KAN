"""
多输出(原始尺度 + 多步) LSTM-KAN 的 SHAP 特征重要性分析。

模型一次输出 H*K = len(HORIZONS)*5 个值（horizon-major：output o -> horizon=HORIZONS[o//K], target=k=o%K）。
shap 的 GradientExplainer.shap_values 对多输出返回“长度 H*K 的列表”（或堆叠数组），
本脚本逐输出取切片，按 (horizon, target) 分别汇总特征级 / feature-lag 级重要性与 best-lag(供 PDP)。

依赖：shap 必须能正常 import（Bayesian 环境 shap 导入崩溃，需先修复；shap 延迟到 main 内导入）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_DIR = PROJECT_ROOT / "base_model"
BEST_OUTPUT_DIR = BASE_MODEL_DIR / "multi_output_model" / "grid_search_outputs_multi"
DATA_DIR = BASE_MODEL_DIR / "data"
OUTPUT_DIR = PROJECT_ROOT / "SHAP" / "shap_lstm_kan_multi_outputs"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base_model.common.data_utils import (  # noqa: E402
    get_raw_feature_cols,
    load_data_splits,
    prepare_multi_horizon_data,
)
from base_model.common.seed_utils import set_seed  # noqa: E402
from base_model.multi_output_model.models import (  # noqa: E402
    HORIZONS,
    KAN,
    SELF_FEATURE_MAP,
    TARGET_COLS_ORDER,
    build_model,
)


MODEL_NAME = "LSTM_KAN"
MODEL_STEM = "lstm_kan"
BACKGROUND_SAMPLE_SIZE = 100
EXPLAINED_SAMPLE_SIZE = 300
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_FILES = {
    "feature_full": OUTPUT_DIR / "shap_feature_importance_full.csv",
    "feature_no_self": OUTPUT_DIR / "shap_feature_importance_no_self.csv",
    "lag_full": OUTPUT_DIR / "shap_lag_importance_full.csv",
    "lag_no_self": OUTPUT_DIR / "shap_lag_importance_no_self.csv",
    "best_lag_full": OUTPUT_DIR / "shap_feature_best_lag_full.csv",
    "best_lag_no_self": OUTPUT_DIR / "shap_feature_best_lag_no_self.csv",
}

GROUP_KEYS = ["target", "horizon"]


def load_best_info() -> dict:
    path = BEST_OUTPUT_DIR / "best_params" / f"best_params_{MODEL_STEM}.json"
    if not path.exists():
        raise FileNotFoundError(f"未找到多输出 LSTM-KAN 最优参数: {path}。请先运行 grid_search_multi_output.py。")
    info = json.loads(path.read_text(encoding="utf-8"))
    for field in ["feature_cols", "target_cols", "best_hyperparameters", "horizons"]:
        if field not in info:
            raise KeyError(f"{path} 缺少必要字段: {field}")
    return info


def load_model_weights(model, path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def sample_indices(n_samples, sample_size, rng):
    if n_samples <= sample_size:
        return np.arange(n_samples)
    return np.sort(rng.choice(n_samples, size=sample_size, replace=False))


def normalize_multi_shap(raw, expected_single, n_outputs):
    """规整为长度 n_outputs 的列表，每个 (n_explained, lookback, n_features)。"""
    if isinstance(raw, (list, tuple)):
        if len(raw) != n_outputs:
            return None
        out = []
        for arr in raw:
            arr = np.asarray(arr)
            if arr.shape == expected_single:
                out.append(arr)
            elif arr.shape == (*expected_single, 1):
                out.append(arr[..., 0])
            else:
                return None
        return out
    arr = np.asarray(raw)
    if arr.shape == (*expected_single, n_outputs):
        return [arr[..., o] for o in range(n_outputs)]
    if arr.shape == (n_outputs, *expected_single):
        return [arr[o] for o in range(n_outputs)]
    if n_outputs == 1 and arr.shape == expected_single:
        return [arr]
    return None


def compute_multi_shap_values(model, x_train, x_test, n_outputs, device, rng, shap_module):
    bg_idx = sample_indices(len(x_train), BACKGROUND_SAMPLE_SIZE, rng)
    exp_idx = sample_indices(len(x_test), EXPLAINED_SAMPLE_SIZE, rng)
    background = torch.from_numpy(x_train[bg_idx]).float().to(device)
    explained = torch.from_numpy(x_test[exp_idx]).float().to(device)
    expected_single = tuple(explained.shape)
    errors = []
    for name, cls in (("GradientExplainer", shap_module.GradientExplainer), ("DeepExplainer", shap_module.DeepExplainer)):
        try:
            explainer = cls(model, background)
            raw = explainer.shap_values(explained)
            per = normalize_multi_shap(raw, expected_single, n_outputs)
            if per is None:
                shape = f"list[{len(raw)}]" if isinstance(raw, (list, tuple)) else np.asarray(raw).shape
                raise ValueError(f"返回形状 {shape}，无法对齐到 {n_outputs}×{expected_single}")
            return per, name, len(bg_idx), len(exp_idx)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
    raise RuntimeError("GradientExplainer 和 DeepExplainer 均失败: " + " | ".join(errors))


def descending_ranks(values):
    values = np.asarray(values, dtype=float)
    return values.argsort()[::-1].argsort() + 1


def summarize_feature_importance(target, horizon, feature_cols, shap_values):
    mean_abs = np.abs(shap_values).mean(axis=(0, 1))
    total = float(mean_abs.sum())
    normalized = mean_abs / total if total > 0 else np.zeros_like(mean_abs)
    ranks = descending_ranks(mean_abs)
    return [
        {"target": target, "horizon": horizon, "feature": f, "mean_abs_shap": float(mean_abs[i]),
         "normalized_importance": float(normalized[i]), "rank": int(ranks[i])}
        for i, f in enumerate(feature_cols)
    ]


def summarize_lag_importance(target, horizon, feature_cols, shap_values):
    lag_importance = np.abs(shap_values).mean(axis=0)  # (lookback, n_features)
    lookback = lag_importance.shape[0]
    global_total = float(lag_importance.sum())
    global_ranks = descending_ranks(lag_importance.reshape(-1)).reshape(lookback, -1)
    rows = []
    for fi, f in enumerate(feature_cols):
        col = lag_importance[:, fi]
        feat_total = float(col.sum())
        lag_ranks = descending_ranks(col)
        for ti in range(lookback):
            lag = lookback - ti
            v = float(lag_importance[ti, fi])
            rows.append({
                "target": target, "horizon": horizon, "feature": f, "lag": int(lag), "time_index": int(ti),
                "mean_abs_shap": v, "normalized_importance": float(v / global_total) if global_total > 0 else 0.0,
                "normalized_importance_within_feature": float(v / feat_total) if feat_total > 0 else 0.0,
                "lag_rank_within_feature": int(lag_ranks[ti]), "rank_within_target": int(global_ranks[ti, fi]),
            })
    return rows


def rerank_feature(df):
    if df.empty:
        return df.copy()
    groups = []
    for _, g in df.groupby(GROUP_KEYS, sort=False):
        g = g.sort_values(["mean_abs_shap", "feature"], ascending=[False, True]).copy()
        total = float(g["mean_abs_shap"].sum())
        g["normalized_importance"] = g["mean_abs_shap"] / total if total > 0 else 0.0
        g["rank"] = np.arange(1, len(g) + 1)
        groups.append(g)
    return pd.concat(groups, ignore_index=True)


def rerank_lag(df):
    if df.empty:
        return df.copy()
    groups = []
    for _, g in df.groupby(GROUP_KEYS, sort=False):
        g = g.copy()
        total = float(g["mean_abs_shap"].sum())
        g["normalized_importance"] = g["mean_abs_shap"] / total if total > 0 else 0.0
        g = g.sort_values(["mean_abs_shap", "feature", "lag"], ascending=[False, True, True])
        g["rank_within_target"] = np.arange(1, len(g) + 1)
        groups.append(g)
    return pd.concat(groups, ignore_index=True)


def build_best_lag_summary(feature_df, lag_df):
    best = lag_df.loc[lag_df["lag_rank_within_feature"] == 1].copy().rename(columns={
        "lag": "best_lag", "time_index": "best_lag_time_index", "mean_abs_shap": "best_lag_mean_abs_shap",
        "normalized_importance": "best_lag_normalized_importance",
        "normalized_importance_within_feature": "best_lag_normalized_importance_within_feature",
        "rank_within_target": "best_lag_rank_within_target",
    })
    meta = feature_df.rename(columns={"mean_abs_shap": "feature_mean_abs_shap",
                                      "normalized_importance": "feature_normalized_importance", "rank": "feature_rank"})
    summary = best.merge(meta, on=["target", "horizon", "feature"], how="left")
    cols = ["target", "horizon", "feature", "feature_rank", "feature_mean_abs_shap", "feature_normalized_importance",
            "best_lag", "best_lag_time_index", "best_lag_mean_abs_shap", "best_lag_normalized_importance",
            "best_lag_normalized_importance_within_feature", "best_lag_rank_within_target"]
    return summary[cols].sort_values(["target", "horizon", "feature_rank", "feature"]).reset_index(drop=True)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"已保存: {path}", flush=True)


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)

    try:
        import shap
    except ImportError as exc:
        raise ImportError(f"未成功 import shap，请先修复 Bayesian 环境的 shap（导入崩溃）。原始错误: {exc}")
    if KAN is None:
        raise ImportError("未安装 efficient-kan 包，请先运行: pip install efficient-kan")

    train_df, val_df, test_df, data_dir, date_col = load_data_splits(DATA_DIR)
    get_raw_feature_cols(train_df, date_col)
    best_info = load_best_info()
    feature_cols, target_cols = best_info["feature_cols"], best_info["target_cols"]
    params = best_info["best_hyperparameters"]
    horizons = best_info["horizons"]
    K = len(TARGET_COLS_ORDER)

    data = prepare_multi_horizon_data(train_df, val_df, test_df, feature_cols, target_cols, params["lookback"], horizons, date_col)
    x_train, x_test = data["train"]["X"], data["test"]["X"]
    output_dim = data["train"]["y"].shape[1]  # H*K

    model = build_model(MODEL_NAME, len(feature_cols), params, output_dim)
    model = load_model_weights(model, BEST_OUTPUT_DIR / "best_models" / f"best_{MODEL_STEM}.pt", device)
    print(f"数据目录: {data_dir} | device={device} | features={len(feature_cols)} | outputs={output_dim} | horizons={horizons}", flush=True)

    rng = np.random.default_rng(SEED)
    per_output, explainer_name, n_bg, n_exp = compute_multi_shap_values(model, x_train, x_test, output_dim, device, rng, shap)
    print(f"explainer={explainer_name} background={n_bg} explained={n_exp}", flush=True)

    feature_rows, lag_rows = [], []
    for o in range(output_dim):
        h_idx, k = o // K, o % K
        target, horizon = TARGET_COLS_ORDER[k], horizons[h_idx]
        sv = per_output[o]  # (n_explained, lookback, n_features)
        feature_rows.extend(summarize_feature_importance(target, horizon, feature_cols, sv))
        lag_rows.extend(summarize_lag_importance(target, horizon, feature_cols, sv))

    feature_full = rerank_feature(pd.DataFrame(feature_rows))
    lag_full = rerank_lag(pd.DataFrame(lag_rows))

    self_df = pd.DataFrame({"target": TARGET_COLS_ORDER, "self_feature": [SELF_FEATURE_MAP[t] for t in TARGET_COLS_ORDER]})
    feature_no_self = feature_full.merge(self_df, on="target", how="left")
    feature_no_self = rerank_feature(feature_no_self.loc[feature_no_self["feature"] != feature_no_self["self_feature"]].drop(columns=["self_feature"]))
    lag_no_self = lag_full.merge(self_df, on="target", how="left")
    lag_no_self = rerank_lag(lag_no_self.loc[lag_no_self["feature"] != lag_no_self["self_feature"]].drop(columns=["self_feature"]))

    save_csv(feature_full, OUTPUT_FILES["feature_full"])
    save_csv(feature_no_self, OUTPUT_FILES["feature_no_self"])
    save_csv(lag_full, OUTPUT_FILES["lag_full"])
    save_csv(lag_no_self, OUTPUT_FILES["lag_no_self"])
    save_csv(build_best_lag_summary(feature_full, lag_full), OUTPUT_FILES["best_lag_full"])
    save_csv(build_best_lag_summary(feature_no_self, lag_no_self), OUTPUT_FILES["best_lag_no_self"])
    print(f"\n完成多输出多步 SHAP 分析。输出目录: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
