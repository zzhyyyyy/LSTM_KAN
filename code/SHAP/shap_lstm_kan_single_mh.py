"""
单输出·多步 LSTM-KAN 的 SHAP 特征重要性分析。

每个目标各一个单输出模型（输出该目标的 t+1/t+2/t+3 共 H 个）。逐目标做 SHAP，
每个目标的 H 个输出对应 (target, horizon)。复用 shap_lstm_kan_multi 的纯函数。

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
BEST_DIR = BASE_MODEL_DIR / "single_output_mh_model" / "grid_search_outputs_single_mh"
DATA_DIR = BASE_MODEL_DIR / "data"
OUTPUT_DIR = PROJECT_ROOT / "SHAP" / "shap_lstm_kan_single_mh_outputs"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "SHAP") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "SHAP"))

from base_model.common.data_utils import get_raw_feature_cols, load_data_splits, prepare_multi_horizon_data  # noqa: E402
from base_model.common.seed_utils import set_seed  # noqa: E402
from base_model.multi_output_model.models import HORIZONS, KAN, SELF_FEATURE_MAP, TARGET_COLS_ORDER, TARGET_MAP_RAW, build_model  # noqa: E402

from shap_lstm_kan_multi import (  # noqa: E402
    OUTPUT_FILES as _MULTI_FILES,  # 仅借用文件名模板结构（此处自定义路径）
    build_best_lag_summary,
    compute_multi_shap_values,
    load_model_weights,
    rerank_feature,
    rerank_lag,
    save_csv,
    summarize_feature_importance,
    summarize_lag_importance,
)


MODEL_STEM = "lstm_kan"
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


def main():
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
    feature_cols_global = get_raw_feature_cols(train_df, date_col)
    rng = np.random.default_rng(SEED)

    feature_rows, lag_rows = [], []
    for target in TARGET_COLS_ORDER:
        bp = BEST_DIR / "best_params" / target / f"best_params_{MODEL_STEM}.json"
        if not bp.exists():
            raise FileNotFoundError(f"未找到 {target} 单输出 LSTM-KAN 最优参数: {bp}。请先运行 single_output_mh_model/grid_search_single_mh.py。")
        info = json.loads(bp.read_text(encoding="utf-8"))
        feature_cols = info["feature_cols"]
        params = info["best_hyperparameters"]
        data = prepare_multi_horizon_data(train_df, val_df, test_df, feature_cols, [info["target_col"]], params["lookback"], info["horizons"], date_col)
        output_dim = len(HORIZONS)
        model = build_model("LSTM_KAN", len(feature_cols), params, output_dim)
        model = load_model_weights(model, BEST_DIR / "best_models" / target / f"best_{MODEL_STEM}.pt", device)

        per_output, explainer_name, n_bg, n_exp = compute_multi_shap_values(
            model, data["train"]["X"], data["test"]["X"], output_dim, device, rng, shap)
        print(f"[{target}] explainer={explainer_name} outputs={output_dim} bg={n_bg} exp={n_exp}", flush=True)
        for o in range(output_dim):  # o -> horizon
            horizon = HORIZONS[o]
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
    print(f"\n完成单输出·多步 SHAP 分析。输出目录: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
