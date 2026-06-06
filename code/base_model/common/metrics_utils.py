from __future__ import annotations

import numpy as np


EPS = 1e-12


def _valid_pair(obs, pred) -> tuple[np.ndarray, np.ndarray]:
    obs_arr = np.asarray(obs, dtype=float).reshape(-1)
    pred_arr = np.asarray(pred, dtype=float).reshape(-1)
    mask = np.isfinite(obs_arr) & np.isfinite(pred_arr)
    return obs_arr[mask], pred_arr[mask]


def rmse(obs, pred) -> float:
    obs_arr, pred_arr = _valid_pair(obs, pred)
    if obs_arr.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((pred_arr - obs_arr) ** 2)))


def mae(obs, pred) -> float:
    obs_arr, pred_arr = _valid_pair(obs, pred)
    if obs_arr.size == 0:
        return np.nan
    return float(np.mean(np.abs(pred_arr - obs_arr)))


def nrmse(obs, pred) -> float:
    obs_arr, pred_arr = _valid_pair(obs, pred)
    if obs_arr.size == 0:
        return np.nan
    obs_mean = np.mean(obs_arr)
    if abs(obs_mean) < EPS:
        return np.nan
    return float(rmse(obs_arr, pred_arr) / obs_mean)


def nmae(obs, pred) -> float:
    obs_arr, pred_arr = _valid_pair(obs, pred)
    if obs_arr.size == 0:
        return np.nan
    obs_mean = np.mean(obs_arr)
    if abs(obs_mean) < EPS:
        return np.nan
    return float(mae(obs_arr, pred_arr) / obs_mean)


def r2(obs, pred) -> float:
    obs_arr, pred_arr = _valid_pair(obs, pred)
    if obs_arr.size == 0:
        return np.nan
    denom = np.sum((obs_arr - np.mean(obs_arr)) ** 2)
    if denom < EPS:
        # 观测值零方差时 R2 无定义，返回 nan 避免误导。
        return np.nan
    return float(1.0 - np.sum((pred_arr - obs_arr) ** 2) / denom)


def nse(obs, pred) -> float:
    obs_arr, pred_arr = _valid_pair(obs, pred)
    if obs_arr.size == 0:
        return np.nan
    denom = np.sum((obs_arr - np.mean(obs_arr)) ** 2)
    if denom < EPS:
        # NSE 与 R2 同样依赖观测方差，零方差时无定义。
        return np.nan
    return float(1.0 - np.sum((pred_arr - obs_arr) ** 2) / denom)


def kge(obs, pred) -> float:
    obs_arr, pred_arr = _valid_pair(obs, pred)
    if obs_arr.size < 2:
        return np.nan

    obs_std = np.std(obs_arr)
    pred_std = np.std(pred_arr)
    obs_mean = np.mean(obs_arr)
    pred_mean = np.mean(pred_arr)

    # 零方差或均值接近 0 时，相关系数/alpha/beta 无稳定定义。
    if obs_std < EPS or pred_std < EPS or abs(obs_mean) < EPS:
        return np.nan

    corr = np.corrcoef(obs_arr, pred_arr)[0, 1]
    alpha = pred_std / obs_std
    beta = pred_mean / obs_mean
    if not np.isfinite(corr) or not np.isfinite(alpha) or not np.isfinite(beta):
        return np.nan

    return float(1.0 - np.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def compute_metrics(obs, pred) -> dict[str, float]:
    """单目标一维指标，输入必须是原始尺度数据。"""
    return {
        "nRMSE": nrmse(obs, pred),
        "nMAE": nmae(obs, pred),
        "NSE": nse(obs, pred),
        "KGE": kge(obs, pred),
    }
