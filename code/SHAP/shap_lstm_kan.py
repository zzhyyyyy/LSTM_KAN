"""
LSTM-KAN 单输出模型 SHAP 特征重要性分析。

从 base_model/single_output_model/grid_search_outputs 加载每个目标的
clean data 网格搜索最优 LSTM-KAN 参数与权重，计算：
1. 特征级 SHAP 重要性；
2. 30 个 lag 上的 feature-lag SHAP 重要性；
3. 每个特征 SHAP 最大的 lag，便于后续 PDP 绘图直接选择对应 lag。
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import shap
except ImportError:
    shap = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_DIR = PROJECT_ROOT / "base_model"
SINGLE_OUTPUT_DIR = BASE_MODEL_DIR / "single_output_model"
BEST_OUTPUT_DIR = SINGLE_OUTPUT_DIR / "grid_search_outputs"
DATA_DIR = BASE_MODEL_DIR / "data"
OUTPUT_DIR = PROJECT_ROOT / "SHAP" / "shap_lstm_kan_outputs"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base_model.common.data_utils import (  # noqa: E402
    get_clean_feature_cols,
    load_data_splits,
    prepare_single_target_data,
)
from base_model.common.seed_utils import set_seed  # noqa: E402
from base_model.single_output_model.grid_search_single_output import (  # noqa: E402
    HORIZON,
    KAN,
    build_model,
)


TARGET_MAP = {
    "Green_Algae": "log_Green_Algae",
    "Cyanobacteria": "log_Cyanobacteria",
    "Diatoms": "log_Diatoms",
    "Cryptophyta": "log_Cryptophyta",
    "Algae_Sum": "log_Algae_Sum",
}
TARGETS_TO_RUN = list(TARGET_MAP.keys())

MODEL_NAME = "LSTM_KAN"
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


def best_params_path(target: str) -> Path:
    return BEST_OUTPUT_DIR / "best_params" / target / "best_params_lstm_kan.json"


def best_model_path(target: str) -> Path:
    return BEST_OUTPUT_DIR / "best_models" / target / "best_lstm_kan.pt"


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"未找到{description}: {path}")


def load_best_info(target: str) -> dict:
    params_path = best_params_path(target)
    model_path = best_model_path(target)
    require_file(params_path, "LSTM-KAN clean data 网格搜索最优参数文件")
    require_file(model_path, "LSTM-KAN clean data 网格搜索最优权重文件")

    best_info = json.loads(params_path.read_text(encoding="utf-8"))
    params = best_info.get("used_hyperparameters") or best_info.get("best_hyperparameters")
    if params is None:
        raise KeyError(f"最优参数文件缺少 used_hyperparameters/best_hyperparameters: {params_path}")

    feature_cols = best_info.get("feature_cols")
    if not feature_cols:
        raise KeyError(f"最优参数文件缺少 feature_cols: {params_path}")

    return {
        "params_path": params_path,
        "model_path": model_path,
        "params": params,
        "target_col": best_info.get("target_col", TARGET_MAP[target]),
        "feature_cols": list(feature_cols),
    }


def load_model_weights(model: torch.nn.Module, path: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def sample_indices(n_samples: int, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    if n_samples <= sample_size:
        return np.arange(n_samples)
    return np.sort(rng.choice(n_samples, size=sample_size, replace=False))


def normalize_shap(raw, expected_shape: tuple[int, ...]) -> np.ndarray | None:
    arr = raw[0] if isinstance(raw, list) else raw
    arr = np.asarray(arr)
    if arr.shape == expected_shape:
        return arr
    if arr.shape == (*expected_shape, 1):
        return arr[..., 0]
    if arr.shape == (1, *expected_shape):
        return arr[0]
    return None


def compute_shap_values(
    model: torch.nn.Module,
    x_train: np.ndarray,
    x_test: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str, int, int]:
    """优先使用 GradientExplainer；失败时回退到 DeepExplainer。"""
    bg_idx = sample_indices(len(x_train), BACKGROUND_SAMPLE_SIZE, rng)
    exp_idx = sample_indices(len(x_test), EXPLAINED_SAMPLE_SIZE, rng)
    background = torch.from_numpy(x_train[bg_idx]).float().to(device)
    explained = torch.from_numpy(x_test[exp_idx]).float().to(device)
    expected_shape = tuple(explained.shape)

    errors: list[str] = []
    for explainer_name, explainer_class in (
        ("GradientExplainer", shap.GradientExplainer),
        ("DeepExplainer", shap.DeepExplainer),
    ):
        try:
            explainer = explainer_class(model, background)
            raw_values = explainer.shap_values(explained)
            values = normalize_shap(raw_values, expected_shape)
            if values is None:
                raise ValueError(f"返回形状 {np.asarray(raw_values).shape}，期望 {expected_shape}")
            return values, explainer_name, len(bg_idx), len(exp_idx)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{explainer_name}: {exc}")

    raise RuntimeError("GradientExplainer 和 DeepExplainer 均失败: " + " | ".join(errors))


def descending_ranks(values: np.ndarray) -> np.ndarray:
    """返回降序排名，最大值排名为 1。"""
    values = np.asarray(values, dtype=float)
    return values.argsort()[::-1].argsort() + 1


def summarize_feature_importance(target: str, feature_cols: list[str], shap_values: np.ndarray) -> list[dict]:
    abs_values = np.abs(shap_values)
    mean_abs = abs_values.mean(axis=(0, 1))
    total = float(mean_abs.sum())
    normalized = mean_abs / total if total > 0 else np.zeros_like(mean_abs)
    ranks = descending_ranks(mean_abs)

    rows = []
    for feature_idx, feature in enumerate(feature_cols):
        rows.append(
            {
                "target": target,
                "feature": feature,
                "mean_abs_shap": float(mean_abs[feature_idx]),
                "normalized_importance": float(normalized[feature_idx]),
                "rank": int(ranks[feature_idx]),
            }
        )
    return rows


def summarize_lag_importance(target: str, feature_cols: list[str], shap_values: np.ndarray) -> list[dict]:
    """
    输出 feature-lag 粒度的 SHAP 重要性。

    shap_values 的时间维度顺序与输入窗口一致：time_index=0 是最早时刻，
    time_index=lookback-1 是最近时刻。为了后续 PDP 标注清晰，这里转换为
    lag=1 表示最近时刻，lag=lookback 表示最早时刻。
    """
    abs_values = np.abs(shap_values)
    lag_importance = abs_values.mean(axis=0)  # [lookback, n_features]
    lookback, n_features = lag_importance.shape
    global_total = float(lag_importance.sum())
    global_ranks = descending_ranks(lag_importance.reshape(-1)).reshape(lookback, n_features)

    rows = []
    for feature_idx, feature in enumerate(feature_cols):
        feature_lag_values = lag_importance[:, feature_idx]
        feature_total = float(feature_lag_values.sum())
        lag_ranks = descending_ranks(feature_lag_values)

        for time_index in range(lookback):
            lag = lookback - time_index
            value = float(lag_importance[time_index, feature_idx])
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "lag": int(lag),
                    "time_index": int(time_index),
                    "mean_abs_shap": value,
                    "normalized_importance": float(value / global_total) if global_total > 0 else 0.0,
                    "normalized_importance_within_feature": (
                        float(value / feature_total) if feature_total > 0 else 0.0
                    ),
                    "lag_rank_within_feature": int(lag_ranks[time_index]),
                    "rank_within_target": int(global_ranks[time_index, feature_idx]),
                }
            )
    return rows


def rerank_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    ranked_groups = []
    for _, group in df.groupby("target", sort=False):
        group = group.sort_values(["mean_abs_shap", "feature"], ascending=[False, True]).copy()
        total = float(group["mean_abs_shap"].sum())
        group["normalized_importance"] = group["mean_abs_shap"] / total if total > 0 else 0.0
        group["rank"] = np.arange(1, len(group) + 1)
        ranked_groups.append(group)
    return pd.concat(ranked_groups, ignore_index=True)


def rerank_lag_importance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    ranked_groups = []
    for _, group in df.groupby("target", sort=False):
        group = group.copy()
        total = float(group["mean_abs_shap"].sum())
        group["normalized_importance"] = group["mean_abs_shap"] / total if total > 0 else 0.0
        group = group.sort_values(["mean_abs_shap", "feature", "lag"], ascending=[False, True, True])
        group["rank_within_target"] = np.arange(1, len(group) + 1)
        ranked_groups.append(group)
    return pd.concat(ranked_groups, ignore_index=True)


def build_best_lag_summary(feature_df: pd.DataFrame, lag_df: pd.DataFrame) -> pd.DataFrame:
    best_lag = lag_df.loc[lag_df["lag_rank_within_feature"] == 1].copy()
    best_lag = best_lag.rename(
        columns={
            "lag": "best_lag",
            "time_index": "best_lag_time_index",
            "mean_abs_shap": "best_lag_mean_abs_shap",
            "normalized_importance": "best_lag_normalized_importance",
            "normalized_importance_within_feature": "best_lag_normalized_importance_within_feature",
            "rank_within_target": "best_lag_rank_within_target",
        }
    )

    feature_meta = feature_df.rename(
        columns={
            "mean_abs_shap": "feature_mean_abs_shap",
            "normalized_importance": "feature_normalized_importance",
            "rank": "feature_rank",
        }
    )

    summary = best_lag.merge(feature_meta, on=["target", "feature"], how="left")
    columns = [
        "target",
        "feature",
        "feature_rank",
        "feature_mean_abs_shap",
        "feature_normalized_importance",
        "best_lag",
        "best_lag_time_index",
        "best_lag_mean_abs_shap",
        "best_lag_normalized_importance",
        "best_lag_normalized_importance_within_feature",
        "best_lag_rank_within_target",
    ]
    return summary[columns].sort_values(["target", "feature_rank", "feature"]).reset_index(drop=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"已保存: {path}", flush=True)


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)

    if shap is None:
        raise ImportError("未安装 shap 包，请先运行: pip install shap")
    if KAN is None:
        raise ImportError("未安装 efficient-kan 包，请先运行: pip install efficient-kan")

    train_df, val_df, test_df, data_dir, date_col = load_data_splits(DATA_DIR)
    clean_feature_cols = get_clean_feature_cols(train_df, date_col)
    print(f"数据目录: {data_dir} | device={device} | clean features={len(clean_feature_cols)}", flush=True)

    rng = np.random.default_rng(SEED)
    feature_rows: list[dict] = []
    lag_rows: list[dict] = []

    for target in TARGETS_TO_RUN:
        print(f"\n[{target}]", flush=True)
        try:
            best_info = load_best_info(target)
            params = dict(best_info["params"])
            feature_cols = list(best_info["feature_cols"])
            target_col = best_info["target_col"]
            lookback = int(params.get("lookback", 30))

            # 明确验证 clean-feature 列，避免误用旧的全特征模型或旧结果目录。
            missing_from_clean = [col for col in feature_cols if col not in clean_feature_cols]
            if missing_from_clean:
                raise ValueError(
                    f"{target} 的网格搜索 feature_cols 含有当前 clean data 中不存在的列: "
                    f"{missing_from_clean}。请检查最优参数文件: {best_info['params_path']}"
                )
            if feature_cols != clean_feature_cols:
                extra_clean = [col for col in clean_feature_cols if col not in feature_cols]
                print(
                    f"WARNING: {target} 的网格搜索 feature_cols 与当前 get_clean_feature_cols "
                    "不完全一致，将按网格搜索保存的列顺序构造输入以匹配模型权重。"
                    f" extra_clean={extra_clean}",
                    file=sys.stderr,
                )

            data = prepare_single_target_data(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_cols=feature_cols,
                target_col=target_col,
                lookback=lookback,
                horizon=HORIZON,
                date_col=date_col,
            )
            x_train = data["train"]["X"]
            x_test = data["test"]["X"]

            model = build_model(MODEL_NAME, len(feature_cols), params)
            model = load_model_weights(model, best_info["model_path"], device)
            print(f"  train={x_train.shape} test={x_test.shape} lookback={lookback}", flush=True)

            shap_values, explainer_name, n_bg, n_exp = compute_shap_values(
                model, x_train, x_test, device, rng
            )
            print(
                f"  explainer={explainer_name} shap_shape={shap_values.shape} "
                f"background={n_bg} explained={n_exp}",
                flush=True,
            )

            feature_rows.extend(summarize_feature_importance(target, feature_cols, shap_values))
            lag_rows.extend(summarize_lag_importance(target, feature_cols, shap_values))

        except Exception as exc:  # noqa: BLE001
            print(f"  错误: {type(exc).__name__}: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)

    if not feature_rows or not lag_rows:
        raise RuntimeError("没有成功生成任何 SHAP 结果，请检查模型、数据和依赖。")

    feature_full = rerank_feature_importance(pd.DataFrame(feature_rows))
    lag_full = rerank_lag_importance(pd.DataFrame(lag_rows))

    self_targets = pd.Series(TARGET_MAP, name="self_feature").rename_axis("target").reset_index()
    feature_no_self = feature_full.merge(self_targets, on="target", how="left")
    feature_no_self = feature_no_self.loc[feature_no_self["feature"] != feature_no_self["self_feature"]]
    feature_no_self = rerank_feature_importance(feature_no_self.drop(columns=["self_feature"]))

    lag_no_self = lag_full.merge(self_targets, on="target", how="left")
    lag_no_self = lag_no_self.loc[lag_no_self["feature"] != lag_no_self["self_feature"]]
    lag_no_self = rerank_lag_importance(lag_no_self.drop(columns=["self_feature"]))

    best_lag_full = build_best_lag_summary(feature_full, lag_full)
    best_lag_no_self = build_best_lag_summary(feature_no_self, lag_no_self)

    save_csv(feature_full, OUTPUT_FILES["feature_full"])
    save_csv(feature_no_self, OUTPUT_FILES["feature_no_self"])
    save_csv(lag_full, OUTPUT_FILES["lag_full"])
    save_csv(lag_no_self, OUTPUT_FILES["lag_no_self"])
    save_csv(best_lag_full, OUTPUT_FILES["best_lag_full"])
    save_csv(best_lag_no_self, OUTPUT_FILES["best_lag_no_self"])

    print("\n完成 LSTM-KAN SHAP lag 重要性分析。", flush=True)
    print(f"输出目录: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
