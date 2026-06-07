from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATE_CANDIDATES = ("date", "Date")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_date_col(df: pd.DataFrame) -> str | None:
    for col in DATE_CANDIDATES:
        if col in df.columns:
            return col
    return None


def sort_by_date_if_needed(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    date_col = find_date_col(df)
    if date_col is None:
        return df.reset_index(drop=True), None

    sorted_df = df.copy()
    sorted_df[date_col] = pd.to_datetime(sorted_df[date_col], errors="coerce")
    sorted_df = sorted_df.sort_values(date_col).reset_index(drop=True)
    return sorted_df, date_col


def load_data_splits(
    data_dir: Path,
    train_name: str = "train_model_input.csv",
    val_name: str = "val_model_input.csv",
    test_name: str = "test_model_input.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, str | None]:
    """从 base_model/data 等指定目录读取 train/val/test。"""
    paths = [data_dir / train_name, data_dir / val_name, data_dir / test_name]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"未找到数据文件: {missing}")

    train_df, train_date_col = sort_by_date_if_needed(pd.read_csv(paths[0]))
    val_df, val_date_col = sort_by_date_if_needed(pd.read_csv(paths[1]))
    test_df, test_date_col = sort_by_date_if_needed(pd.read_csv(paths[2]))

    date_col = train_date_col or val_date_col or test_date_col
    return train_df, val_df, test_df, data_dir, date_col


def get_clean_feature_cols(df: pd.DataFrame, date_col: str | None = None) -> list[str]:
    """
    构建去冗余后的输入特征列。
    原则：
    1. 如果变量存在 log 版本，则只保留 log 版本；
    2. 如果变量没有 log 版本，则保留原始版本；
    3. 不使用 date 列；
    4. 不同时输入原始变量和 log 变量，避免信息重复。
    """
    preferred_cols = [
        "log_Green_Algae",
        "log_Cyanobacteria",
        "log_Diatoms",
        "log_Cryptophyta",
        "log_Algae_Sum",
        "log_Turbidity",
        "log_DIP",
        "log_TP",
        "log_precipitation",
        "log_NPR",
        "pH",
        "WT",
        "DIN",
        "TN",
    ]

    return [col for col in preferred_cols if col in df.columns and col != date_col]


def get_raw_feature_cols(df: pd.DataFrame, date_col: str | None = None) -> list[str]:
    """
    原始尺度输入特征（不做 log 变换，符合研究进展文档"基础模型在原始尺度下完成"）。
    4 个底层藻类 + 9 个环境/水质变量，共 13 个；按文档排除 Algae_Sum
    （总藻不作为输入，避免信息泄漏与冗余）。顺序固定，供模型权重对齐。
    """
    preferred_cols = [
        "Green_Algae",
        "Cyanobacteria",
        "Diatoms",
        "Cryptophyta",
        "pH",
        "Turbidity",
        "WT",
        "precipitation",
        "DIN",
        "DIP",
        "TN",
        "TP",
        "NPR",
    ]
    return [col for col in preferred_cols if col in df.columns and col != date_col]


def get_log_feature_cols(df: pd.DataFrame, date_col: str | None = None) -> list[str]:
    """
    方案A：log1p 尺度输入特征。对偏态变量(藻类、浊度、DIP、TP、降水、氮磷比)用 log1p，
    pH/WT/DIN/TN 保留原始；按文档**排除 Algae_Sum**（不含 log_Algae_Sum）。共 13 个，顺序固定。
    等价于 get_clean_feature_cols 去掉 log_Algae_Sum。
    """
    preferred_cols = [
        "log_Green_Algae",
        "log_Cyanobacteria",
        "log_Diatoms",
        "log_Cryptophyta",
        "log_Turbidity",
        "log_DIP",
        "log_TP",
        "log_precipitation",
        "log_NPR",
        "pH",
        "WT",
        "DIN",
        "TN",
    ]
    return [col for col in preferred_cols if col in df.columns and col != date_col]


def infer_feature_cols(
    train_df: pd.DataFrame,
    explicit_feature_cols: Iterable[str] | None = None,
    date_col: str | None = None,
) -> list[str]:
    if explicit_feature_cols is not None:
        missing = [col for col in explicit_feature_cols if col not in train_df.columns]
        if missing:
            raise ValueError(f"FEATURE_COLS 中存在缺失列: {missing}")
        return list(explicit_feature_cols)

    # 默认使用去冗余后的显式特征列表，避免同时输入原始变量和 log 变量。
    # 多变量模型仍然使用多个藻类和环境因子，但每个变量只保留一种尺度。
    return get_clean_feature_cols(train_df, date_col)


def _check_required_columns(df: pd.DataFrame, cols: list[str], split_name: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{split_name} 缺失必要列: {missing}")
    na_count = int(df[cols].isna().sum().sum())
    if na_count > 0:
        raise ValueError(f"{split_name} 在特征/目标列中存在缺失值，共 {na_count} 个")


def _get_dates(df: pd.DataFrame, date_col: str | None) -> np.ndarray | None:
    if date_col is None or date_col not in df.columns:
        return None
    return df[date_col].to_numpy()


def build_windows(
    x_all: np.ndarray,
    y_all: np.ndarray,
    dates_all: np.ndarray | None,
    lookback: int,
    horizon: int,
    target_start_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    x_windows, y_values, y_dates = [], [], []
    max_start = len(x_all) - lookback - horizon + 1
    for start in range(max_start):
        target_idx = start + lookback + horizon - 1
        if target_idx < target_start_index:
            continue
        x_windows.append(x_all[start : start + lookback])
        y_values.append(y_all[target_idx])
        if dates_all is not None:
            y_dates.append(dates_all[target_idx])

    if not x_windows:
        raise ValueError("窗口样本为空，请检查 LOOKBACK/HORIZON 与数据长度")

    x_arr = np.asarray(x_windows, dtype=np.float32)
    # reshape(len, -1) 保留目标列数：单目标 y_all 为 (T,1) 时结果仍是 (N,1)，
    # 与原 reshape(-1,1) 字节一致；多目标 y_all 为 (T,K) 时结果为 (N,K)。
    y_arr = np.asarray(y_values, dtype=np.float32).reshape(len(y_values), -1)
    date_arr = np.asarray(y_dates) if dates_all is not None else None
    return x_arr, y_arr, date_arr


def build_windows_multi_horizon(
    x_all: np.ndarray,
    y_all: np.ndarray,
    dates_all: np.ndarray | None,
    lookback: int,
    horizons: list[int],
    target_start_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    多步（多 horizon）滚动窗口：每个窗口同时给出 horizons 指定的若干步目标。
    y_all 形状 (T, K)。返回 y 形状 (N, H*K)，列序为 horizon-major：
    index = h_idx*K + k，即 y[:, h_idx*K + k] = 第 horizons[h_idx] 步、第 k 个目标。
    dates 形状 (N, H)，对应每个窗口各 horizon 的目标日期，便于按步落盘。
    """
    horizons = list(horizons)
    max_h = max(horizons)
    x_windows, y_values, y_dates = [], [], []
    max_start = len(x_all) - lookback - max_h + 1
    for start in range(max_start):
        first_target_idx = start + lookback + horizons[0] - 1
        if first_target_idx < target_start_index:
            continue
        x_windows.append(x_all[start : start + lookback])
        steps = [y_all[start + lookback + h - 1] for h in horizons]  # H × (K,)
        y_values.append(np.stack(steps, axis=0).reshape(-1))  # (H*K,)
        if dates_all is not None:
            y_dates.append([dates_all[start + lookback + h - 1] for h in horizons])  # (H,)

    if not x_windows:
        raise ValueError("多步窗口样本为空，请检查 LOOKBACK/HORIZONS 与数据长度")

    x_arr = np.asarray(x_windows, dtype=np.float32)
    y_arr = np.asarray(y_values, dtype=np.float32)  # (N, H*K)
    date_arr = np.asarray(y_dates) if dates_all is not None else None  # (N, H)
    return x_arr, y_arr, date_arr


def prepare_multi_target_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    lookback: int,
    horizon: int,
    date_col: str | None = None,
) -> dict:
    """
    多目标窗口构造：对 target_cols（K 列）联合 fit 一个 StandardScaler，
    y 形状为 (N, K)，列序严格等于传入的 target_cols。其余上下文拼接逻辑与
    单目标版本完全一致（验证/测试集前接 lookback 步历史，避免边界丢样本）。

    单目标只是 K=1 的特例：prepare_single_target_data 直接委托到本函数。
    """
    target_cols = list(target_cols)
    cols = list(dict.fromkeys(feature_cols + target_cols))
    _check_required_columns(train_df, cols, "train")
    _check_required_columns(val_df, cols, "val")
    _check_required_columns(test_df, cols, "test")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    train_x = x_scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=float))
    val_x = x_scaler.transform(val_df[feature_cols].to_numpy(dtype=float))
    test_x = x_scaler.transform(test_df[feature_cols].to_numpy(dtype=float))

    train_y = y_scaler.fit_transform(train_df[target_cols].to_numpy(dtype=float))
    val_y = y_scaler.transform(val_df[target_cols].to_numpy(dtype=float))
    test_y = y_scaler.transform(test_df[target_cols].to_numpy(dtype=float))

    train_dates = _get_dates(train_df, date_col)
    val_dates = _get_dates(val_df, date_col)
    test_dates = _get_dates(test_df, date_col)

    train_split = build_windows(train_x, train_y, train_dates, lookback, horizon, 0)

    val_ctx_x = train_x[-lookback:]
    val_ctx_y = train_y[-lookback:]
    val_ctx_dates = train_dates[-lookback:] if train_dates is not None else None
    val_all_dates = (
        np.concatenate([val_ctx_dates, val_dates])
        if val_ctx_dates is not None and val_dates is not None
        else None
    )
    val_split = build_windows(
        np.vstack([val_ctx_x, val_x]),
        np.vstack([val_ctx_y, val_y]),
        val_all_dates,
        lookback,
        horizon,
        target_start_index=lookback,
    )

    history_x = np.vstack([train_x, val_x])[-lookback:]
    history_y = np.vstack([train_y, val_y])[-lookback:]
    if train_dates is not None and val_dates is not None and test_dates is not None:
        history_dates = np.concatenate([train_dates, val_dates])[-lookback:]
        test_all_dates = np.concatenate([history_dates, test_dates])
    else:
        test_all_dates = None
    test_split = build_windows(
        np.vstack([history_x, test_x]),
        np.vstack([history_y, test_y]),
        test_all_dates,
        lookback,
        horizon,
        target_start_index=lookback,
    )

    return {
        "train": {"X": train_split[0], "y": train_split[1], "dates": train_split[2]},
        "val": {"X": val_split[0], "y": val_split[1], "dates": val_split[2]},
        "test": {"X": test_split[0], "y": test_split[1], "dates": test_split[2]},
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "target_cols": target_cols,
    }


def prepare_multi_horizon_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    lookback: int,
    horizons: list[int],
    date_col: str | None = None,
) -> dict:
    """
    多目标 + 多步（multi-horizon）数据构造，原始尺度（不做 log）。
    对 target_cols（K 列）联合 fit 一个 StandardScaler（仅标准化、无 log），
    每个 horizon 的 K 维目标用同一套均值/方差标准化（跨 horizon 共享，量纲一致）。
    y 形状 (N, H*K)，列序 horizon-major（index = h_idx*K + k）。
    反变换请用 inverse_targets（只反标准化、不 expm1）。
    """
    target_cols = list(target_cols)
    horizons = list(horizons)
    cols = list(dict.fromkeys(feature_cols + target_cols))
    _check_required_columns(train_df, cols, "train")
    _check_required_columns(val_df, cols, "val")
    _check_required_columns(test_df, cols, "test")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    train_x = x_scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=float))
    val_x = x_scaler.transform(val_df[feature_cols].to_numpy(dtype=float))
    test_x = x_scaler.transform(test_df[feature_cols].to_numpy(dtype=float))

    train_y = y_scaler.fit_transform(train_df[target_cols].to_numpy(dtype=float))
    val_y = y_scaler.transform(val_df[target_cols].to_numpy(dtype=float))
    test_y = y_scaler.transform(test_df[target_cols].to_numpy(dtype=float))

    train_dates = _get_dates(train_df, date_col)
    val_dates = _get_dates(val_df, date_col)
    test_dates = _get_dates(test_df, date_col)

    train_split = build_windows_multi_horizon(train_x, train_y, train_dates, lookback, horizons, 0)

    val_ctx_x = train_x[-lookback:]
    val_ctx_y = train_y[-lookback:]
    val_ctx_dates = train_dates[-lookback:] if train_dates is not None else None
    val_all_dates = (
        np.concatenate([val_ctx_dates, val_dates])
        if val_ctx_dates is not None and val_dates is not None
        else None
    )
    val_split = build_windows_multi_horizon(
        np.vstack([val_ctx_x, val_x]),
        np.vstack([val_ctx_y, val_y]),
        val_all_dates,
        lookback,
        horizons,
        target_start_index=lookback,
    )

    history_x = np.vstack([train_x, val_x])[-lookback:]
    history_y = np.vstack([train_y, val_y])[-lookback:]
    if train_dates is not None and val_dates is not None and test_dates is not None:
        history_dates = np.concatenate([train_dates, val_dates])[-lookback:]
        test_all_dates = np.concatenate([history_dates, test_dates])
    else:
        test_all_dates = None
    test_split = build_windows_multi_horizon(
        np.vstack([history_x, test_x]),
        np.vstack([history_y, test_y]),
        test_all_dates,
        lookback,
        horizons,
        target_start_index=lookback,
    )

    return {
        "train": {"X": train_split[0], "y": train_split[1], "dates": train_split[2]},
        "val": {"X": val_split[0], "y": val_split[1], "dates": val_split[2]},
        "test": {"X": test_split[0], "y": test_split[1], "dates": test_split[2]},
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "target_cols": target_cols,
        "horizons": horizons,
        "n_targets": len(target_cols),
        "n_horizons": len(horizons),
    }


def prepare_single_target_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lookback: int,
    horizon: int,
    date_col: str | None = None,
) -> dict:
    """单目标版本，等价于 prepare_multi_target_data 的 K=1 特例（y 形状 (N,1)）。"""
    return prepare_multi_target_data(
        train_df,
        val_df,
        test_df,
        feature_cols,
        [target_col],
        lookback,
        horizon,
        date_col,
    )


def inverse_log_target(y_scaled, y_scaler: StandardScaler) -> np.ndarray:
    y_log = y_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).reshape(-1)
    return np.expm1(y_log)


def inverse_log_targets(y_scaled_matrix, y_scaler: StandardScaler) -> np.ndarray:
    """
    多目标反变换：对 (N, K) 标准化矩阵先 inverse_transform 再 expm1，返回 (N, K)。
    列序与 fit 时的 target_cols 一致。不要用于单列 1D 数组（请用 inverse_log_target）。
    """
    y_scaled = np.asarray(y_scaled_matrix, dtype=float)
    if y_scaled.ndim != 2:
        raise ValueError(f"inverse_log_targets 需要二维 (N,K) 矩阵，收到 ndim={y_scaled.ndim}")
    y_log = y_scaler.inverse_transform(y_scaled)
    return np.expm1(y_log)


def inverse_targets(y_scaled_matrix, y_scaler: StandardScaler) -> np.ndarray:
    """
    原始尺度多目标反变换：只做反标准化，不做 expm1（目标本身就是原始值）。
    输入 (N, K) -> 返回 (N, K)，列序与 fit 时的 target_cols 一致。
    """
    y_scaled = np.asarray(y_scaled_matrix, dtype=float)
    if y_scaled.ndim != 2:
        raise ValueError(f"inverse_targets 需要二维 (N,K) 矩阵，收到 ndim={y_scaled.ndim}")
    return y_scaler.inverse_transform(y_scaled)


def append_csv_row(path: Path, row: dict, columns: list[str]) -> None:
    ensure_dir(path.parent)
    safe_row = {col: row.get(col, np.nan) for col in columns}
    exists = path.exists()
    pd.DataFrame([safe_row], columns=columns).to_csv(
        path, mode="a", header=not exists, index=False, encoding="utf-8-sig"
    )


def load_completed_trials(path: Path, target: str, model_name: str) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["target", "model_name", "trial_id"])
    mask = (df["target"] == target) & (df["model_name"] == model_name)
    return set(df.loc[mask, "trial_id"].astype(str))
