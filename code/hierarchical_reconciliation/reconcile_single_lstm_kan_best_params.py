from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BASE_MODEL_DIR = PROJECT_DIR / "base_model"
SINGLE_OUTPUT_DIR = BASE_MODEL_DIR / "single_output_model"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(BASE_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_MODEL_DIR))
if str(SINGLE_OUTPUT_DIR) not in sys.path:
    sys.path.insert(0, str(SINGLE_OUTPUT_DIR))

from base_model.common.data_utils import (  # noqa: E402
    ensure_dir,
    get_clean_feature_cols,
    load_data_splits,
    prepare_single_target_data,
)
from base_model.common.metrics_utils import compute_metrics  # noqa: E402
from base_model.common.seed_utils import set_seed  # noqa: E402
from base_model.single_output_model.grid_search_single_output import (  # noqa: E402
    DATA_DIR,
    HORIZON,
    KAN,
    MODEL_DISPLAY,
    MODEL_STEM,
    SEED,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
    build_model,
    evaluate_on_split,
    metric_text,
    params_text,
)


# =========================
# 路径与层次结构配置
# =========================
BEST_OUTPUT_DIR = SINGLE_OUTPUT_DIR / "grid_search_outputs"
OUTPUT_DIR = SCRIPT_DIR / "reconciliation_outputs"

MODEL_NAME = "LSTM_KAN"
MODELS_TO_RUN = [MODEL_NAME]

TOP_TARGET = "Algae_Sum"
BOTTOM_TARGETS = ["Green_Algae", "Cyanobacteria", "Diatoms", "Cryptophyta"]
ALL_TARGETS = [TOP_TARGET, *BOTTOM_TARGETS]

METHODS = ["Base", "BU", "TD", "OLS", "WLS", "MinT"]
PRINT_METHODS = ["BU", "TD", "OLS", "WLS", "MinT"]
PREDICTION_METHOD_COLUMNS = {
    "Base": "base_pred",
    "BU": "BU_pred",
    "TD": "TD_pred",
    "OLS": "OLS_pred",
    "WLS": "WLS_pred",
    "MinT": "MinT_pred",
}
PREDICTION_COLUMNS = [
    "date",
    "sample_index",
    "model",
    "target",
    "y_true",
    *PREDICTION_METHOD_COLUMNS.values(),
]
METRIC_COLUMNS = ["model", "target", "method", "nRMSE", "nMAE", "NSE", "KGE"]

MINT_SHRINKAGE = 0.1
MINT_RIDGE = 1e-6


def best_params_path(target: str, model_name: str) -> Path:
    return (
        BEST_OUTPUT_DIR
        / "best_params"
        / target
        / f"best_params_{MODEL_STEM[model_name]}.json"
    )


def best_model_path(target: str, model_name: str) -> Path:
    return (
        BEST_OUTPUT_DIR
        / "best_models"
        / target
        / f"best_{MODEL_STEM[model_name]}.pt"
    )


def load_best_info(target: str, model_name: str) -> dict:
    path = best_params_path(target, model_name)
    if not path.exists():
        raise FileNotFoundError(f"未找到最佳参数文件: {path}")
    info = json.loads(path.read_text(encoding="utf-8"))
    if "best_hyperparameters" not in info:
        if "used_hyperparameters" not in info:
            raise KeyError(f"{path} 缺少 best_hyperparameters 或 used_hyperparameters。")
        info["best_hyperparameters"] = info["used_hyperparameters"]
    for field in ["target_col", "feature_cols", "best_hyperparameters"]:
        if field not in info:
            raise KeyError(f"{path} 缺少必要字段: {field}")
    info["base_model_source_dir"] = str(BEST_OUTPUT_DIR)
    return info


def require_best_outputs() -> None:
    if not BEST_OUTPUT_DIR.exists():
        raise FileNotFoundError(f"未找到基础模型网格搜索输出目录: {BEST_OUTPUT_DIR}")
    for target in ALL_TARGETS:
        for path in [best_params_path(target, MODEL_NAME), best_model_path(target, MODEL_NAME)]:
            if not path.exists():
                raise FileNotFoundError(f"未找到 LSTM-KAN 网格搜索最佳文件: {path}")


def build_summing_matrix() -> np.ndarray:
    """构造汇总矩阵 S，将四个底层藻类映射为 [总藻, 四个底层藻类]。"""
    bottom_count = len(BOTTOM_TARGETS)
    top_row = np.ones((1, bottom_count), dtype=float)
    bottom_rows = np.eye(bottom_count, dtype=float)
    return np.vstack([top_row, bottom_rows])


def td_proportions(train_df: pd.DataFrame) -> np.ndarray:
    """仅使用训练集真实值计算 top-down 底层平均比例，避免验证集或测试集泄漏。"""
    bottom_values = train_df[BOTTOM_TARGETS].to_numpy(dtype=float)
    top_values = train_df[[TOP_TARGET]].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        proportions = bottom_values / top_values
    proportions[~np.isfinite(proportions)] = np.nan
    mean_proportions = np.nanmean(proportions, axis=0)
    if not np.all(np.isfinite(mean_proportions)) or mean_proportions.sum() <= 0:
        raise ValueError("训练集无法计算有效 TD 比例。")
    return mean_proportions / mean_proportions.sum()


def projection_matrix(
    summing_matrix: np.ndarray,
    covariance: np.ndarray | None = None,
) -> np.ndarray:
    """
    构造层次协调投影矩阵：
    y_tilde = S G y_hat
    G = (S' W^-1 S)^-1 S' W^-1

    covariance 的含义：
    - None：OLS，等价于 W = I；
    - 对角矩阵：WLS，使用结构尺度近似误差方差；
    - 完整协方差矩阵：MinT，使用验证集基础预测残差估计。
    """
    series_count = summing_matrix.shape[0]
    if covariance is None:
        w_inv = np.eye(series_count)
    else:
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (series_count, series_count):
            raise ValueError(
                f"协方差矩阵形状错误，应为 {(series_count, series_count)}，实际为 {covariance.shape}"
            )
        w_inv = np.linalg.pinv(covariance)

    middle = summing_matrix.T @ w_inv @ summing_matrix
    gain = np.linalg.pinv(middle) @ summing_matrix.T @ w_inv
    return summing_matrix @ gain


def estimate_mint_shrink_cov(
    errors: np.ndarray,
    shrinkage: float = MINT_SHRINKAGE,
    ridge: float = MINT_RIDGE,
) -> np.ndarray:
    """
    使用验证集基础预测残差估计 MinT-shrink 协方差矩阵。

    errors shape = [n_series, n_samples]，n_series 对应
    [Algae_Sum, Green_Algae, Cyanobacteria, Diatoms, Cryptophyta]。
    不能使用测试集残差估计协方差，否则会把测试集真实值泄漏到协调过程。
    """
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 2:
        raise ValueError(f"errors 应为二维矩阵，实际维度为 {errors.ndim}")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(f"shrinkage 应位于 [0, 1]，实际为 {shrinkage}")
    if not np.all(np.isfinite(errors)):
        raise ValueError("用于估计 MinT 协方差的验证集残差包含非有限值。")

    series_count, sample_count = errors.shape
    if sample_count < 2:
        raise ValueError("用于估计 MinT 协方差的样本数不足。")

    sample_cov = np.cov(errors)
    diag_cov = np.diag(np.diag(sample_cov))
    cov = (1.0 - shrinkage) * sample_cov + shrinkage * diag_cov
    cov += ridge * np.eye(series_count)
    return cov


def reconcile_predictions(
    test_base_matrix: np.ndarray,
    td_props: np.ndarray,
    mint_cov: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    输入矩阵行为 [Algae_Sum, 四个底层藻类]，列为测试样本。
    BU 由底层相加得到顶层；TD 用训练集比例拆分顶层预测；
    OLS/WLS/MinT 共用 projection_matrix，仅协方差矩阵不同。
    """
    summing_matrix = build_summing_matrix()
    bottom_base = test_base_matrix[1:, :]
    top_base = test_base_matrix[0, :]

    bu = np.vstack([bottom_base.sum(axis=0, keepdims=True), bottom_base])
    td_bottom = td_props.reshape(-1, 1) @ top_base.reshape(1, -1)
    td = np.vstack([top_base.reshape(1, -1), td_bottom])

    ols = projection_matrix(summing_matrix) @ test_base_matrix
    structural_cov = np.diag(summing_matrix.sum(axis=1))
    wls = projection_matrix(summing_matrix, structural_cov) @ test_base_matrix
    mint = projection_matrix(summing_matrix, mint_cov) @ test_base_matrix

    return {
        "Base": test_base_matrix,
        "BU": bu,
        "TD": td,
        "OLS": ols,
        "WLS": wls,
        "MinT": mint,
    }


def load_best_model_outputs(
    target: str,
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str | None,
    clean_feature_cols: list[str],
    device: torch.device,
) -> dict:
    best_info = load_best_info(target, model_name)
    missing_from_clean = [col for col in best_info["feature_cols"] if col not in clean_feature_cols]
    if missing_from_clean:
        raise ValueError(
            f"{target}/{MODEL_DISPLAY[model_name]} 的网格搜索 feature_cols 含有当前 clean data "
            f"中不存在的列: {missing_from_clean}。请确认 grid_search_outputs 来自 clean data 网格搜索。"
        )
    if best_info["feature_cols"] != clean_feature_cols:
        extra_clean = [col for col in clean_feature_cols if col not in best_info["feature_cols"]]
        print(
            f"WARNING: {target}/{MODEL_DISPLAY[model_name]} 的网格搜索 feature_cols 与当前 "
            "get_clean_feature_cols 不完全一致，将按网格搜索保存的列顺序构造输入以匹配模型权重。"
            f" extra_clean={extra_clean}",
            file=sys.stderr,
        )
    params = best_info["best_hyperparameters"]
    data = prepare_single_target_data(
        train_df,
        val_df,
        test_df,
        best_info["feature_cols"],
        best_info["target_col"],
        params["lookback"],
        HORIZON,
        date_col,
    )

    set_seed(SEED)
    model = build_model(model_name, data["train"]["X"].shape[-1], params)
    state_path = best_model_path(target, model_name)
    if not state_path.exists():
        raise FileNotFoundError(f"未找到最佳模型权重文件: {state_path}")
    model.load_state_dict(torch.load(state_path, map_location="cpu"))

    val_true, val_pred, val_metrics = evaluate_on_split(
        model, data["val"], data["y_scaler"], params["batch_size"], device
    )
    test_true, test_pred, test_metrics = evaluate_on_split(
        model, data["test"], data["y_scaler"], params["batch_size"], device
    )

    print(
        f"[{MODEL_DISPLAY[model_name]}][{target}] params={params_text(params)} | "
        f"best_epoch={best_info.get('best_epoch', 'NA')} | "
        f"val nRMSE={metric_text(val_metrics['nRMSE'])} | "
        f"test nRMSE={metric_text(test_metrics['nRMSE'])}"
    )
    return {
        "target_col": best_info["target_col"],
        "params": params,
        "val_dates": data["val"]["dates"],
        "test_dates": data["test"]["dates"],
        "val_true": val_true,
        "val_base_pred": val_pred,
        "test_true": test_true,
        "test_base_pred": test_pred,
    }


def validate_split_dates(outputs: dict[str, dict], split: str) -> np.ndarray | None:
    date_key = f"{split}_dates"
    dates = outputs[ALL_TARGETS[0]][date_key]
    for target in ALL_TARGETS[1:]:
        other_dates = outputs[target][date_key]
        if dates is None or other_dates is None:
            if dates is not None or other_dates is not None:
                raise ValueError(f"{split} 集日期信息不一致。")
            continue
        if len(dates) != len(other_dates) or np.any(
            pd.to_datetime(dates) != pd.to_datetime(other_dates)
        ):
            raise ValueError(f"各目标 {split} 集日期不一致，无法进行层次协调。")
    return dates


def align_outputs(outputs: dict[str, dict]) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    """对齐五个目标的验证集和测试集输出矩阵，MinT 协方差只使用验证集矩阵。"""
    validate_split_dates(outputs, "val")
    test_dates = validate_split_dates(outputs, "test")
    arrays = {
        "val_true": np.vstack([outputs[target]["val_true"] for target in ALL_TARGETS]),
        "val_base": np.vstack([outputs[target]["val_base_pred"] for target in ALL_TARGETS]),
        "test_true": np.vstack([outputs[target]["test_true"] for target in ALL_TARGETS]),
        "test_base": np.vstack([outputs[target]["test_base_pred"] for target in ALL_TARGETS]),
    }
    return test_dates, arrays


def metric_row(model_label: str, target: str, method: str, y_true, y_pred) -> dict:
    metrics = compute_metrics(y_true, y_pred)
    return {
        "model": model_label,
        "target": target,
        "method": method,
        "nRMSE": metrics["nRMSE"],
        "nMAE": metrics["nMAE"],
        "NSE": metrics["NSE"],
        "KGE": metrics["KGE"],
    }


def append_model_outputs(
    model_name: str,
    dates,
    true_matrix: np.ndarray,
    reconciled: dict[str, np.ndarray],
    prediction_store: dict[str, list[dict]],
    metric_store: dict[str, list[dict]],
    summary_rows: list[dict],
) -> None:
    model_label = MODEL_DISPLAY[model_name]
    sample_index = np.arange(true_matrix.shape[1])

    for target_idx, target in enumerate(ALL_TARGETS):
        row_values = {
            "sample_index": sample_index,
            "model": model_label,
            "target": target,
            "y_true": true_matrix[target_idx],
        }
        for method, column in PREDICTION_METHOD_COLUMNS.items():
            row_values[column] = reconciled[method][target_idx]

        rows = pd.DataFrame(row_values)
        if dates is not None:
            rows.insert(0, "date", pd.to_datetime(dates).strftime("%Y-%m-%d"))
        else:
            rows.insert(0, "date", np.nan)
        prediction_store[target].extend(rows[PREDICTION_COLUMNS].to_dict("records"))

        for method in METHODS:
            row = metric_row(
                model_label,
                target,
                method,
                true_matrix[target_idx],
                reconciled[method][target_idx],
            )
            metric_store[target].append(row)
            summary_rows.append(row)


def save_outputs(
    output_dir: Path,
    prediction_store: dict[str, list[dict]],
    metric_store: dict[str, list[dict]],
    summary_rows: list[dict],
) -> Path:
    for target in ALL_TARGETS:
        target_dir = output_dir / target
        ensure_dir(target_dir)
        pd.DataFrame(prediction_store[target], columns=PREDICTION_COLUMNS).to_csv(
            target_dir / f"{target}_reconciled_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(metric_store[target], columns=METRIC_COLUMNS).to_csv(
            target_dir / f"{target}_reconciliation_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary_dir = output_dir / "summary"
    ensure_dir(summary_dir)
    summary_path = summary_dir / "reconciliation_metrics.csv"
    summary_df = pd.DataFrame(summary_rows, columns=METRIC_COLUMNS)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_path


def main() -> None:
    set_seed(SEED)
    require_best_outputs()
    if MODEL_NAME in MODELS_TO_RUN and KAN is None:
        raise ImportError("未检测到 efficient-kan，请先安装 efficient-kan 后再运行 LSTM-KAN。")

    output_dir = OUTPUT_DIR
    ensure_dir(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df, actual_data_dir, date_col = load_data_splits(
        DATA_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV
    )
    clean_feature_cols = get_clean_feature_cols(train_df, date_col)
    proportions = td_proportions(train_df)
    print(f"Data: {actual_data_dir} | device={device} | clean features={len(clean_feature_cols)}")
    print(f"Best LSTM-KAN source: {BEST_OUTPUT_DIR}")
    print(f"TD proportions from train set: {dict(zip(BOTTOM_TARGETS, proportions.round(6)))}")

    prediction_store = {target: [] for target in ALL_TARGETS}
    metric_store = {target: [] for target in ALL_TARGETS}
    summary_rows = []

    for model_name in MODELS_TO_RUN:
        print(f"\nCurrent model: {MODEL_DISPLAY[model_name]}")
        outputs = {
            target: load_best_model_outputs(
                target, model_name, train_df, val_df, test_df, date_col, clean_feature_cols, device
            )
            for target in ALL_TARGETS
        }
        dates, arrays = align_outputs(outputs)
        val_errors = arrays["val_true"] - arrays["val_base"]
        mint_cov = estimate_mint_shrink_cov(val_errors, shrinkage=MINT_SHRINKAGE)
        reconciled = reconcile_predictions(arrays["test_base"], proportions, mint_cov)
        append_model_outputs(
            model_name,
            dates,
            arrays["test_true"],
            reconciled,
            prediction_store,
            metric_store,
            summary_rows,
        )
        print(
            f"Completed model: {MODEL_DISPLAY[model_name]} | "
            f"methods={', '.join(METHODS)}"
        )

    summary_path = save_outputs(output_dir, prediction_store, metric_store, summary_rows)
    print(f"\nCompleted reconciliation methods: {', '.join(PRINT_METHODS)}")
    print(f"Output directory: {output_dir}")
    print(f"Summary metrics: {summary_path}")


if __name__ == "__main__":
    main()
