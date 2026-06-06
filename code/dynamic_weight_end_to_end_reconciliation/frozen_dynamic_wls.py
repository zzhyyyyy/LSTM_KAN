from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
    inverse_log_target,
    load_data_splits,
    prepare_single_target_data,
)
from base_model.common.metrics_utils import compute_metrics  # noqa: E402
from base_model.common.seed_utils import set_seed  # noqa: E402
from base_model.common.train_utils import save_loss_history  # noqa: E402
from base_model.single_output_model.grid_search_single_output import (  # noqa: E402
    DATA_DIR,
    EARLY_STOPPING_PATIENCE,
    HORIZON,
    KAN,
    MAX_EPOCHS,
    MODEL_DISPLAY,
    MODEL_STEM,
    SEED,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
    build_model,
    iter_grid,
    json_safe,
    metric_text,
    params_text,
)


# =========================
# 路径与运行配置
# =========================
BEST_OUTPUT_DIR = SINGLE_OUTPUT_DIR / "grid_search_outputs"
OUTPUT_DIR = SCRIPT_DIR / "frozen_dynamic_wls_outputs"

# 本模块固定以 LSTM-KAN 作为唯一基础预报模型，不再比较 LSTM-FC/LSTM-MLP。
MODEL_NAME = "LSTM_KAN"
MODELS_TO_RUN = [MODEL_NAME]

# 默认使用已有 LSTM-KAN 最优权重初始化，并冻结五个基础分支，只训练动态权重 MLP。
INITIALIZE_FROM_BEST_MODEL = True
FREEZE_BASE_BRANCHES = True
JOINT_BATCH_SIZE = None
WEIGHT_EPS = 1e-6
SOLVE_RIDGE = 1e-6
GRAD_CLIP_NORM = 5.0
EPS = 1e-12
CONSISTENCY_TOL = 1e-6
CONSISTENCY_LOSS_WEIGHT = 0.0
CLIP_NONNEGATIVE = False

# 专利材料口径提醒：
# 1. 当前脚本只评价 LSTM-KAN 基础预报上叠加 DynamicWLS 后的一致性改善；
# 2. 动态权重协调模块的核心贡献由 D1 到 D2 的一致性提升体现；
# 3. 协调后精度可能略低于基础模型，这是精度-一致性权衡，不应简单视为失败。
MLP_GRID = {
    "hidden_dim": [8, 32, 64],
    "num_layers": [1, 2],
    "dropout": [0.0, 0.1],
    "lr": [1e-3, 5e-4],
}

# 层次结构：Algae_Sum = Green_Algae + Cyanobacteria + Diatoms + Cryptophyta
TOP_TARGET = "Algae_Sum"
BOTTOM_TARGETS = ["Green_Algae", "Cyanobacteria", "Diatoms", "Cryptophyta"]
ALL_TARGETS = [TOP_TARGET, *BOTTOM_TARGETS]
METHODS = ["Base", "DynamicWLS"]
DATASETS = ["train", "val", "test"]

MLP_PARAM_COLUMNS = ["mlp_hidden_dim", "mlp_num_layers", "mlp_dropout", "mlp_lr"]
GRID_RESULT_COLUMNS = [
    "model_name",
    "model_display",
    "trial_id",
    *MLP_PARAM_COLUMNS,
    "joint_batch_size",
    "freeze_base_branches",
    "initialize_from_best_model",
    "best_epoch",
    "train_loss_at_best_epoch",
    "val_loss_at_best_epoch",
    "validation_mean_nRMSE",
    "validation_mean_nMAE",
    "validation_mean_NSE",
    "validation_mean_KGE",
    "validation_D1_mean_abs",
    "validation_D2_mean_abs",
    "validation_D1_max_abs",
    "validation_D2_max_abs",
    "validation_consistency_improvement_mean_pct",
    "validation_consistency_improvement_max_pct",
    "validation_hierarchy_max_abs_error",
]
METRIC_COLUMNS = [
    "model",
    "target",
    "dataset",
    "method",
    "nRMSE",
    "nMAE",
    "NSE",
    "KGE",
    "hierarchy_max_abs_error",
    "D1_mean_abs",
    "D2_mean_abs",
    "consistency_improvement_mean_pct",
]

CONSISTENCY_COLUMNS = [
    "date",
    "dataset",
    "sample_index",
    "model",
    "D1_abs",
    "D2_abs",
    "D1_signed",
    "D2_signed",
    "consistency_improvement_pct",
    "base_top_pred",
    "base_bottom_sum_pred",
    "reconciled_top_pred",
    "reconciled_bottom_sum_pred",
]

CONSISTENCY_SUMMARY_COLUMNS = [
    "model",
    "dataset",
    "D1_mean_abs",
    "D2_mean_abs",
    "D1_median_abs",
    "D2_median_abs",
    "D1_max_abs",
    "D2_max_abs",
    "D1_sum_abs",
    "D2_sum_abs",
    "consistency_improvement_mean_pct",
    "consistency_improvement_median_pct",
    "consistency_improvement_max_pct",
    "consistency_improvement_sum_pct",
]

FINAL_REPORT_COLUMNS = [
    "dataset",
    "target",
    "base_nRMSE",
    "reconciled_nRMSE",
    "base_nMAE",
    "reconciled_nMAE",
    "base_NSE",
    "reconciled_NSE",
    "base_KGE",
    "reconciled_KGE",
    "accuracy_change_nRMSE_pct",
    "accuracy_change_nMAE_pct",
    "accuracy_change_NSE_pct",
    "accuracy_change_KGE_pct",
    "D1_mean_abs",
    "D2_mean_abs",
    "consistency_improvement_mean_pct",
    "D1_max_abs",
    "D2_max_abs",
    "consistency_improvement_max_pct",
]


def build_summing_matrix() -> np.ndarray:
    """构造层次加和矩阵 S，将四个底层节点映射到全层级节点。"""
    bottom_count = len(BOTTOM_TARGETS)
    top_row = np.ones((1, bottom_count), dtype=np.float32)
    bottom_rows = np.eye(bottom_count, dtype=np.float32)
    return np.vstack([top_row, bottom_rows])


def best_params_path(target: str, model_name: str) -> Path:
    if model_name != MODEL_NAME:
        raise ValueError(f"当前动态协调模块只支持 {MODEL_NAME}，收到: {model_name}")
    return (
        BEST_OUTPUT_DIR
        / "best_params"
        / target
        / f"best_params_{MODEL_STEM[model_name]}.json"
    )


def best_model_path(target: str, model_name: str) -> Path:
    if model_name != MODEL_NAME:
        raise ValueError(f"当前动态协调模块只支持 {MODEL_NAME}，收到: {model_name}")
    return (
        BEST_OUTPUT_DIR
        / "best_models"
        / target
        / f"best_{MODEL_STEM[model_name]}.pt"
    )


def load_best_info(target: str, model_name: str) -> dict:
    path = best_params_path(target, model_name)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 {target} 的 LSTM-KAN 网格搜索最优参数，请先完成 "
            f"single_output_model 的 clean data 网格搜索。缺失文件: {path}"
        )
    info = json.loads(path.read_text(encoding="utf-8"))

    # 兼容旧结果中的 used_hyperparameters；这里归一为旧字段名，
    # 让后续 DynamicWLS 训练逻辑继续复用原有读取路径。
    if "best_hyperparameters" not in info:
        if "used_hyperparameters" not in info:
            raise KeyError(f"{path} 缺少 used_hyperparameters，无法构建 LSTM-KAN 分支。")
        info["best_hyperparameters"] = info["used_hyperparameters"]
    info["base_model_source_dir"] = str(BEST_OUTPUT_DIR)
    return info


def output_paths(model_name: str) -> dict[str, Path]:
    stem = MODEL_STEM[model_name]
    return {
        "grid": OUTPUT_DIR / "grid_search" / f"{stem}_dynamic_weight_mlp_grid_results.csv",
        "loss_dir": OUTPUT_DIR / "loss_history" / model_name,
        "best_params": OUTPUT_DIR / "best_params" / f"best_dynamic_weight_{stem}.json",
        "best_model": OUTPUT_DIR / "best_models" / f"best_dynamic_weight_{stem}.pt",
    }


def ensure_output_dirs() -> None:
    for subdir in [
        "grid_search",
        "loss_history",
        "best_params",
        "best_models",
        "metrics",
        "consistency",
        "summary",
    ]:
        ensure_dir(OUTPUT_DIR / subdir)


def mlp_params_for_row(params: dict) -> dict:
    return {
        "mlp_hidden_dim": params["hidden_dim"],
        "mlp_num_layers": params["num_layers"],
        "mlp_dropout": params["dropout"],
        "mlp_lr": params["lr"],
    }


def load_completed_trials(path: Path, model_name: str) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["model_name", "trial_id"])
    return set(df.loc[df["model_name"] == model_name, "trial_id"].astype(str))


def append_csv_row(path: Path, row: dict, columns: list[str]) -> None:
    ensure_dir(path.parent)
    safe_row = {col: row.get(col, np.nan) for col in columns}
    exists = path.exists()
    if exists:
        old_df = pd.read_csv(path)
        if list(old_df.columns) != columns:
            for col in columns:
                if col not in old_df.columns:
                    old_df[col] = np.nan
            old_df = old_df.reindex(columns=columns)
            old_df.to_csv(path, index=False, encoding="utf-8-sig")
    pd.DataFrame([safe_row], columns=columns).to_csv(
        path, mode="a", header=not exists, index=False, encoding="utf-8-sig"
    )


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def backup_stale_file(path: Path, timestamp: str) -> None:
    if not path.exists():
        return
    backup_path = path.with_name(f"{path.stem}.stale_{timestamp}{path.suffix}")
    path.replace(backup_path)
    print(f"WARNING: moved stale output: {path} -> {backup_path}", file=sys.stderr)


def load_best_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    info = json.loads(path.read_text(encoding="utf-8"))
    summary_keys = [
        "validation_mean_nRMSE",
        "validation_D2_mean_abs",
    ]
    summary = {}
    for key in summary_keys:
        best_key = f"best_{key}"
        if best_key not in info:
            return None
        summary[key] = info[best_key]
    return summary


def normalize_saved_base_info(info: dict) -> dict[str, dict] | None:
    saved_infos = info.get("base_best_info")
    if not isinstance(saved_infos, dict):
        return None

    normalized = {}
    for target in ALL_TARGETS:
        if target not in saved_infos:
            return None
        target_info = dict(saved_infos[target])
        if "best_hyperparameters" not in target_info:
            if "used_hyperparameters" not in target_info:
                return None
            target_info["best_hyperparameters"] = target_info["used_hyperparameters"]
        normalized[target] = target_info
    return normalized


def saved_base_matches_current(saved_params_path: Path, current_infos: dict[str, dict]) -> bool:
    if not saved_params_path.exists():
        return True

    saved_info = json.loads(saved_params_path.read_text(encoding="utf-8"))
    saved_infos = normalize_saved_base_info(saved_info)
    if saved_infos is None:
        return False

    # 只比较会影响模型结构和输入顺序的字段，避免因绝对路径变化误判。
    checked_fields = ["target_col", "feature_cols", "best_hyperparameters"]
    for target in ALL_TARGETS:
        for field in checked_fields:
            if saved_infos[target].get(field) != current_infos[target].get(field):
                return False
    return True


def is_close(a: float, b: float, tol: float = CONSISTENCY_TOL) -> bool:
    if not np.isfinite(a) or not np.isfinite(b):
        return False
    return abs(a - b) <= tol


def is_better_trial(current_summary: dict, best_summary: dict | None) -> bool:
    """优先选择层次一致性更好的 trial；D2 已接近 0 时再比较预测精度。"""
    if best_summary is None:
        return True

    current_d2 = float(current_summary.get("validation_D2_mean_abs", float("inf")))
    best_d2 = float(best_summary.get("validation_D2_mean_abs", float("inf")))
    current_nrmse = float(current_summary.get("validation_mean_nRMSE", float("inf")))
    best_nrmse = float(best_summary.get("validation_mean_nRMSE", float("inf")))

    current_consistent = current_d2 <= CONSISTENCY_TOL
    best_consistent = best_d2 <= CONSISTENCY_TOL
    if current_consistent and not best_consistent:
        return True
    if current_consistent and best_consistent:
        return current_nrmse < best_nrmse
    if not current_consistent and not best_consistent:
        if is_close(current_d2, best_d2):
            return current_nrmse < best_nrmse
        return current_d2 < best_d2
    return False


class DynamicWeightMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, input_dim))
        self.net = nn.Sequential(*layers)
        self.positive = nn.Softplus()

    def forward(self, base_pred: torch.Tensor) -> torch.Tensor:
        return self.positive(self.net(base_pred)) + WEIGHT_EPS


class DynamicHierarchicalReconciliationModel(nn.Module):
    def __init__(
        self,
        branches: dict[str, nn.Module],
        target_means: list[float],
        target_scales: list[float],
        mlp_params: dict,
    ):
        super().__init__()
        self.branches = nn.ModuleDict(branches)
        self.weight_mlp = DynamicWeightMLP(
            len(ALL_TARGETS),
            mlp_params["hidden_dim"],
            mlp_params["num_layers"],
            mlp_params["dropout"],
        )
        self.register_buffer("target_means", torch.tensor(target_means, dtype=torch.float32))
        self.register_buffer("target_scales", torch.tensor(target_scales, dtype=torch.float32))
        self.register_buffer("summing_matrix", torch.tensor(build_summing_matrix(), dtype=torch.float32))

    def forward(self, branch_inputs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        scaled_outputs = []
        for target, xb in zip(ALL_TARGETS, branch_inputs):
            scaled_outputs.append(self.branches[target](xb).reshape(-1))

        base_scaled = torch.stack(scaled_outputs, dim=1)
        base_log = base_scaled * self.target_scales.unsqueeze(0) + self.target_means.unsqueeze(0)
        base_pred = torch.expm1(base_log)
        if CLIP_NONNEGATIVE:
            base_pred = torch.clamp(base_pred, min=0.0)
        weights = self.weight_mlp(base_pred)
        reconciled, bottom_reconciled = self.reconcile(base_pred, weights)
        return {
            "base_scaled": base_scaled,
            "base_pred": base_pred,
            "weights": weights,
            "bottom_reconciled": bottom_reconciled,
            "reconciled": reconciled,
        }

    def reconcile(
        self, base_pred: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        动态加权最小二乘协调层：
        bottom = (S.T @ W^-1 @ S)^-1 @ S.T @ W^-1 @ base_pred
        all = S @ bottom

        这里 S 是固定层次加和矩阵，W 是每个样本由 MLP 生成的对角权重矩阵。
        使用 torch.linalg.solve 避免显式求逆，并加入极小 ridge 提升数值稳定性。
        """
        s_matrix = self.summing_matrix
        w_inv = 1.0 / weights
        weighted_s = s_matrix.unsqueeze(0) * w_inv.unsqueeze(-1)
        middle = torch.matmul(s_matrix.T.unsqueeze(0), weighted_s)
        eye = torch.eye(s_matrix.shape[1], device=base_pred.device, dtype=base_pred.dtype)
        middle = middle + SOLVE_RIDGE * eye.unsqueeze(0)
        right = torch.matmul(
            s_matrix.T.unsqueeze(0),
            (w_inv * base_pred).unsqueeze(-1),
        )
        bottom_reconciled = torch.linalg.solve(middle, right).squeeze(-1)
        if CLIP_NONNEGATIVE:
            bottom_reconciled = torch.clamp(bottom_reconciled, min=0.0)
        all_reconciled = torch.matmul(bottom_reconciled, s_matrix.T)
        return all_reconciled, bottom_reconciled


def prepare_all_target_data(
    best_infos: dict[str, dict],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str | None,
    clean_feature_cols: list[str] | None = None,
) -> dict[str, dict]:
    if clean_feature_cols is None:
        clean_feature_cols = get_clean_feature_cols(train_df, date_col)
    prepared = {}
    for target in ALL_TARGETS:
        info = best_infos[target]
        missing_from_clean = [col for col in info["feature_cols"] if col not in clean_feature_cols]
        if missing_from_clean:
            raise ValueError(
                f"{target} 的网格搜索 feature_cols 含有当前 clean data 中不存在的列: "
                f"{missing_from_clean}。请确认 grid_search_outputs 来自 clean data 网格搜索。"
            )
        if info["feature_cols"] != clean_feature_cols:
            extra_clean = [col for col in clean_feature_cols if col not in info["feature_cols"]]
            print(
                f"WARNING: {target} 的网格搜索 feature_cols 与当前 get_clean_feature_cols "
                "不完全一致，将按网格搜索保存的列顺序构造输入以匹配模型权重。"
                f" extra_clean={extra_clean}",
                file=sys.stderr,
            )
        params = info["best_hyperparameters"]
        prepared[target] = prepare_single_target_data(
            train_df,
            val_df,
            test_df,
            info["feature_cols"],
            info["target_col"],
            params["lookback"],
            HORIZON,
            date_col,
        )
    for dataset in DATASETS:
        validate_split_alignment(prepared, dataset)
    return prepared


def validate_split_alignment(prepared: dict[str, dict], dataset: str) -> None:
    base_dates = prepared[ALL_TARGETS[0]][dataset]["dates"]
    base_n = len(prepared[ALL_TARGETS[0]][dataset]["X"])
    for target in ALL_TARGETS[1:]:
        other_n = len(prepared[target][dataset]["X"])
        if other_n != base_n:
            raise ValueError(f"{dataset} 样本数不一致，无法进行端到端协调。")

        other_dates = prepared[target][dataset]["dates"]
        if base_dates is None or other_dates is None:
            if base_dates is not None or other_dates is not None:
                raise ValueError(f"{dataset} 日期信息不一致，无法对齐样本。")
            continue
        if len(base_dates) != len(other_dates) or np.any(
            pd.to_datetime(base_dates) != pd.to_datetime(other_dates)
        ):
            raise ValueError(f"{dataset} 各目标日期不一致，无法进行层次协调。")


def true_matrix(prepared: dict[str, dict], dataset: str) -> np.ndarray:
    values = []
    for target in ALL_TARGETS:
        split = prepared[target][dataset]
        values.append(inverse_log_target(split["y"], prepared[target]["y_scaler"]))
    return np.stack(values, axis=1).astype(np.float32)


def split_dates(prepared: dict[str, dict], dataset: str):
    return prepared[ALL_TARGETS[0]][dataset]["dates"]


def make_joint_loader(
    prepared: dict[str, dict],
    dataset: str,
    batch_size: int,
    shuffle: bool,
    seed: int = SEED,
) -> DataLoader:
    tensors = [
        torch.from_numpy(prepared[target][dataset]["X"]).float()
        for target in ALL_TARGETS
    ]
    tensors.append(torch.from_numpy(true_matrix(prepared, dataset)).float())
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def joint_batch_size(best_infos: dict[str, dict]) -> int:
    if JOINT_BATCH_SIZE is not None:
        return int(JOINT_BATCH_SIZE)
    return min(int(best_infos[target]["best_hyperparameters"]["batch_size"]) for target in ALL_TARGETS)


def build_dynamic_model(
    model_name: str,
    best_infos: dict[str, dict],
    prepared: dict[str, dict],
    mlp_params: dict,
) -> DynamicHierarchicalReconciliationModel:
    branches = {}
    target_means, target_scales = [], []
    for target in ALL_TARGETS:
        params = best_infos[target]["best_hyperparameters"]
        input_dim = prepared[target]["train"]["X"].shape[-1]
        branch = build_model(model_name, input_dim, params)
        if INITIALIZE_FROM_BEST_MODEL:
            state_path = best_model_path(target, model_name)
            if not state_path.exists():
                raise FileNotFoundError(
                    f"未找到 {target} 的 LSTM-KAN 网格搜索最优权重，请先完成 "
                    f"single_output_model 的 clean data 网格搜索。缺失文件: {state_path}"
                )
            branch.load_state_dict(torch.load(state_path, map_location="cpu"))
        branches[target] = branch

        scaler = prepared[target]["y_scaler"]
        target_means.append(float(scaler.mean_[0]))
        target_scales.append(float(scaler.scale_[0]))

    model = DynamicHierarchicalReconciliationModel(branches, target_means, target_scales, mlp_params)
    if FREEZE_BASE_BRANCHES:
        for branch in model.branches.values():
            for param in branch.parameters():
                param.requires_grad = False
    return model


def optimizer_for_model(
    model: DynamicHierarchicalReconciliationModel,
    best_infos: dict[str, dict],
    mlp_params: dict,
) -> torch.optim.Optimizer:
    param_groups = [
        {
            "params": [p for p in model.weight_mlp.parameters() if p.requires_grad],
            "lr": mlp_params["lr"],
        }
    ]
    if not FREEZE_BASE_BRANCHES:
        for target in ALL_TARGETS:
            params = [p for p in model.branches[target].parameters() if p.requires_grad]
            if params:
                branch_lr = best_infos[target]["best_hyperparameters"]["lr"]
                param_groups.append({"params": params, "lr": branch_lr})
    return torch.optim.Adam(param_groups)


def _state_dict_to_cpu(state_dict: dict) -> dict:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def train_dynamic_model(
    model: DynamicHierarchicalReconciliationModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[list[dict], dict, int, float, float]:
    criterion = nn.MSELoss()
    model.to(device)
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    train_loss_at_best = float("inf")
    wait = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_sum, train_n = 0.0, 0
        for batch in train_loader:
            branch_inputs = [item.to(device) for item in batch[:-1]]
            y_true = batch[-1].to(device)
            optimizer.zero_grad()
            output = model(branch_inputs)
            prediction_loss = criterion(output["reconciled"], y_true)
            consistency_residual = output["reconciled"][:, 0] - output["reconciled"][:, 1:].sum(dim=1)
            consistency_loss = torch.mean(consistency_residual**2)
            loss = prediction_loss + CONSISTENCY_LOSS_WEIGHT * consistency_loss
            loss.backward()
            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            train_sum += loss.item() * y_true.size(0)
            train_n += y_true.size(0)
        train_loss = train_sum / max(train_n, 1)

        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                branch_inputs = [item.to(device) for item in batch[:-1]]
                y_true = batch[-1].to(device)
                output = model(branch_inputs)
                prediction_loss = criterion(output["reconciled"], y_true)
                consistency_residual = output["reconciled"][:, 0] - output["reconciled"][:, 1:].sum(dim=1)
                consistency_loss = torch.mean(consistency_residual**2)
                loss = prediction_loss + CONSISTENCY_LOSS_WEIGHT * consistency_loss
                val_sum += loss.item() * y_true.size(0)
                val_n += y_true.size(0)
        val_loss = val_sum / max(val_n, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_epoch = epoch
            train_loss_at_best = train_loss
            best_state = _state_dict_to_cpu(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is None:
        best_state = _state_dict_to_cpu(model.state_dict())
    model.load_state_dict(copy.deepcopy(best_state))
    return history, best_state, best_epoch, train_loss_at_best, best_val_loss


@torch.no_grad()
def predict_dynamic_model(
    model: DynamicHierarchicalReconciliationModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.to(device)
    model.eval()
    stores = {"true": [], "base": [], "reconciled": [], "weights": []}
    for batch in loader:
        branch_inputs = [item.to(device) for item in batch[:-1]]
        y_true = batch[-1].to(device)
        output = model(branch_inputs)
        stores["true"].append(y_true.detach().cpu().numpy())
        stores["base"].append(output["base_pred"].detach().cpu().numpy())
        stores["reconciled"].append(output["reconciled"].detach().cpu().numpy())
        stores["weights"].append(output["weights"].detach().cpu().numpy())
    return {key: np.vstack(value) for key, value in stores.items()}


def hierarchy_residual(values: np.ndarray) -> float:
    residual = values[:, 0] - values[:, 1:].sum(axis=1)
    return float(np.max(np.abs(residual))) if len(residual) else np.nan


def safe_improvement_pct(before: float, after: float) -> float:
    if not np.isfinite(before) or abs(before) <= EPS:
        return np.nan
    return float((before - after) / before * 100.0)


def compute_required_metrics(y_true, y_pred) -> dict[str, float]:
    """兼容旧版 metrics_utils：确保返回 nRMSE、nMAE、NSE、KGE 四个指标。"""
    metrics = compute_metrics(y_true, y_pred)
    if {"nRMSE", "nMAE", "NSE", "KGE"}.issubset(metrics):
        return {
            "nRMSE": metrics["nRMSE"],
            "nMAE": metrics["nMAE"],
            "NSE": metrics["NSE"],
            "KGE": metrics["KGE"],
        }

    true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    valid = np.isfinite(true_arr) & np.isfinite(pred_arr)
    true_arr = true_arr[valid]
    pred_arr = pred_arr[valid]
    if true_arr.size == 0:
        nrmse = np.nan
        nmae = np.nan
    else:
        target_mean = float(np.mean(true_arr))
        if abs(target_mean) <= EPS:
            nrmse = np.nan
            nmae = np.nan
        else:
            nrmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)) / target_mean)
            nmae = float(np.mean(np.abs(pred_arr - true_arr)) / target_mean)

    return {
        "nRMSE": nrmse,
        "nMAE": nmae,
        "NSE": metrics.get("NSE", np.nan),
        "KGE": metrics.get("KGE", np.nan),
    }


def consistency_rows_for_split(
    model_name: str,
    dataset: str,
    dates,
    predictions: dict[str, np.ndarray],
) -> list[dict]:
    model_label = MODEL_DISPLAY[model_name]
    n_samples = predictions["base"].shape[0]
    sample_index = np.arange(n_samples)
    date_values = (
        pd.to_datetime(dates).strftime("%Y-%m-%d")
        if dates is not None
        else np.full(n_samples, np.nan)
    )

    base_top = predictions["base"][:, 0]
    base_bottom_sum = predictions["base"][:, 1:].sum(axis=1)
    reconciled_top = predictions["reconciled"][:, 0]
    reconciled_bottom_sum = predictions["reconciled"][:, 1:].sum(axis=1)
    d1_signed = base_top - base_bottom_sum
    d2_signed = reconciled_top - reconciled_bottom_sum
    d1_abs = np.abs(d1_signed)
    d2_abs = np.abs(d2_signed)
    with np.errstate(divide="ignore", invalid="ignore"):
        improvement = (d1_abs - d2_abs) / d1_abs * 100.0
    improvement[d1_abs <= EPS] = np.nan

    rows = pd.DataFrame(
        {
            "date": date_values,
            "dataset": dataset,
            "sample_index": sample_index,
            "model": model_label,
            "D1_abs": d1_abs,
            "D2_abs": d2_abs,
            "D1_signed": d1_signed,
            "D2_signed": d2_signed,
            "consistency_improvement_pct": improvement,
            "base_top_pred": base_top,
            "base_bottom_sum_pred": base_bottom_sum,
            "reconciled_top_pred": reconciled_top,
            "reconciled_bottom_sum_pred": reconciled_bottom_sum,
        }
    )
    return rows[CONSISTENCY_COLUMNS].to_dict("records")


def summarize_consistency(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    if df.empty:
        return {col: np.nan for col in CONSISTENCY_SUMMARY_COLUMNS}

    d1 = pd.to_numeric(df["D1_abs"], errors="coerce").to_numpy(dtype=float)
    d2 = pd.to_numeric(df["D2_abs"], errors="coerce").to_numpy(dtype=float)
    d1_mean = float(np.nanmean(d1))
    d2_mean = float(np.nanmean(d2))
    d1_median = float(np.nanmedian(d1))
    d2_median = float(np.nanmedian(d2))
    d1_max = float(np.nanmax(d1))
    d2_max = float(np.nanmax(d2))
    d1_sum = float(np.nansum(d1))
    d2_sum = float(np.nansum(d2))
    return {
        "model": df["model"].iloc[0],
        "dataset": df["dataset"].iloc[0],
        "D1_mean_abs": d1_mean,
        "D2_mean_abs": d2_mean,
        "D1_median_abs": d1_median,
        "D2_median_abs": d2_median,
        "D1_max_abs": d1_max,
        "D2_max_abs": d2_max,
        "D1_sum_abs": d1_sum,
        "D2_sum_abs": d2_sum,
        "consistency_improvement_mean_pct": safe_improvement_pct(d1_mean, d2_mean),
        "consistency_improvement_median_pct": safe_improvement_pct(d1_median, d2_median),
        "consistency_improvement_max_pct": safe_improvement_pct(d1_max, d2_max),
        "consistency_improvement_sum_pct": safe_improvement_pct(d1_sum, d2_sum),
    }


def metric_rows_for_split(
    model_name: str,
    dataset: str,
    predictions: dict[str, np.ndarray],
    consistency_summary: dict,
) -> list[dict]:
    model_label = MODEL_DISPLAY[model_name]
    base_residual = hierarchy_residual(predictions["base"])
    reconciled_residual = hierarchy_residual(predictions["reconciled"])
    rows = []
    for target_idx, target in enumerate(ALL_TARGETS):
        for method, key, residual in [
            ("Base", "base", base_residual),
            ("DynamicWLS", "reconciled", reconciled_residual),
        ]:
            metrics = compute_required_metrics(
                predictions["true"][:, target_idx],
                predictions[key][:, target_idx],
            )
            rows.append(
                {
                    "model": model_label,
                    "target": target,
                    "dataset": dataset,
                    "method": method,
                    "nRMSE": metrics["nRMSE"],
                    "nMAE": metrics["nMAE"],
                    "NSE": metrics["NSE"],
                    "KGE": metrics["KGE"],
                    "hierarchy_max_abs_error": residual,
                    "D1_mean_abs": consistency_summary["D1_mean_abs"],
                    "D2_mean_abs": (
                        np.nan if method == "Base" else consistency_summary["D2_mean_abs"]
                    ),
                    "consistency_improvement_mean_pct": (
                        np.nan
                        if method == "Base"
                        else consistency_summary["consistency_improvement_mean_pct"]
                    ),
                }
            )
    return rows


def validation_summary(rows: list[dict], consistency_summaries: list[dict]) -> dict:
    dynamic_rows = [
        row for row in rows if row["dataset"] == "val" and row["method"] == "DynamicWLS"
    ]
    val_consistency = next(row for row in consistency_summaries if row["dataset"] == "val")
    return {
        "validation_mean_nRMSE": float(np.nanmean([row["nRMSE"] for row in dynamic_rows])),
        "validation_mean_nMAE": float(np.nanmean([row["nMAE"] for row in dynamic_rows])),
        "validation_mean_NSE": float(np.nanmean([row["NSE"] for row in dynamic_rows])),
        "validation_mean_KGE": float(np.nanmean([row["KGE"] for row in dynamic_rows])),
        "validation_D1_mean_abs": val_consistency["D1_mean_abs"],
        "validation_D2_mean_abs": val_consistency["D2_mean_abs"],
        "validation_D1_max_abs": val_consistency["D1_max_abs"],
        "validation_D2_max_abs": val_consistency["D2_max_abs"],
        "validation_consistency_improvement_mean_pct": val_consistency[
            "consistency_improvement_mean_pct"
        ],
        "validation_consistency_improvement_max_pct": val_consistency[
            "consistency_improvement_max_pct"
        ],
        "validation_hierarchy_max_abs_error": float(
            np.nanmax([row["hierarchy_max_abs_error"] for row in dynamic_rows])
        ),
    }


def evaluate_model_on_splits(
    model_name: str,
    model: DynamicHierarchicalReconciliationModel,
    prepared: dict[str, dict],
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict], list[dict], list[dict]]:
    metric_rows = []
    consistency_rows = []
    consistency_summary_rows = []
    for dataset in DATASETS:
        loader = make_joint_loader(prepared, dataset, batch_size, False, seed=SEED)
        predictions = predict_dynamic_model(model, loader, device)
        split_consistency_rows = consistency_rows_for_split(
            model_name, dataset, split_dates(prepared, dataset), predictions
        )
        split_consistency_summary = summarize_consistency(split_consistency_rows)
        consistency_rows.extend(split_consistency_rows)
        consistency_summary_rows.append(split_consistency_summary)
        metric_rows.extend(
            metric_rows_for_split(model_name, dataset, predictions, split_consistency_summary)
        )
    return metric_rows, consistency_rows, consistency_summary_rows


def save_best_outputs(
    model_name: str,
    metric_rows: list[dict],
    consistency_rows: list[dict],
    consistency_summary_rows: list[dict],
) -> None:
    stem = MODEL_STEM[model_name]
    metrics_path = OUTPUT_DIR / "metrics" / f"{stem}_dynamic_weight_metrics.csv"
    pd.DataFrame(metric_rows, columns=METRIC_COLUMNS).to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(consistency_rows, columns=CONSISTENCY_COLUMNS).to_csv(
        OUTPUT_DIR / "consistency" / f"{stem}_consistency_by_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(consistency_summary_rows, columns=CONSISTENCY_SUMMARY_COLUMNS).to_csv(
        OUTPUT_DIR / "consistency" / f"{stem}_consistency_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def train_grid_for_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str | None,
    clean_feature_cols: list[str],
    device: torch.device,
) -> tuple[list[dict], list[dict], list[dict]]:
    best_infos = {target: load_best_info(target, model_name) for target in ALL_TARGETS}
    prepared = prepare_all_target_data(
        best_infos, train_df, val_df, test_df, date_col, clean_feature_cols
    )
    batch_size = joint_batch_size(best_infos)
    paths = output_paths(model_name)
    completed = load_completed_trials(paths["grid"], model_name)
    best_summary = load_best_summary(paths["best_params"])
    saved_best_is_stale = not saved_base_matches_current(paths["best_params"], best_infos)
    if saved_best_is_stale:
        print(
            "WARNING: existing DynamicWLS best_params was trained with different base "
            "LSTM-KAN inputs or hyperparameters. Ignoring saved DynamicWLS best model "
            "and rerunning all trials with current clean data grid_search_outputs.",
            file=sys.stderr,
        )
        completed = set()
        best_summary = None
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        for stale_path in [paths["grid"], paths["best_params"], paths["best_model"]]:
            backup_stale_file(stale_path, timestamp)
    if not saved_best_is_stale and paths["best_params"].exists() and best_summary is None:
        print(
            "WARNING: existing dynamic best_params does not contain D2 consistency fields; "
            "rerunning current LSTM-KAN DynamicWLS grid with the consistency-first selector.",
            file=sys.stderr,
        )
        completed = set()
    best_metric_rows = None
    best_consistency_rows = None
    best_consistency_summary_rows = None
    if (
        not saved_best_is_stale
        and best_summary is None
        and paths["best_params"].exists()
        and paths["best_model"].exists()
    ):
        (
            best_metric_rows,
            best_consistency_rows,
            best_consistency_summary_rows,
        ) = evaluate_saved_best_model(model_name, best_infos, prepared, batch_size, device)
        best_summary = validation_summary(best_metric_rows, best_consistency_summary_rows)

    grid = iter_grid(MLP_GRID)
    for idx, mlp_params in enumerate(grid, start=1):
        trial_id = f"trial_{idx:04d}"
        if trial_id in completed:
            continue

        set_seed(SEED)
        model = build_dynamic_model(model_name, best_infos, prepared, mlp_params)
        train_loader = make_joint_loader(prepared, "train", batch_size, True, seed=SEED)
        val_loader = make_joint_loader(prepared, "val", batch_size, False, seed=SEED)
        optimizer = optimizer_for_model(model, best_infos, mlp_params)

        history, best_state, best_epoch, train_loss_best, val_loss_best = train_dynamic_model(
            model, train_loader, val_loader, optimizer, device
        )
        loss_path = paths["loss_dir"] / f"{trial_id}_loss.csv"
        save_loss_history(history, loss_path)

        metric_rows, consistency_rows, consistency_summary_rows = evaluate_model_on_splits(
            model_name, model, prepared, batch_size, device
        )
        summary = validation_summary(metric_rows, consistency_summary_rows)
        row = {
            "model_name": model_name,
            "model_display": MODEL_DISPLAY[model_name],
            "trial_id": trial_id,
            **mlp_params_for_row(mlp_params),
            "joint_batch_size": batch_size,
            "freeze_base_branches": FREEZE_BASE_BRANCHES,
            "initialize_from_best_model": INITIALIZE_FROM_BEST_MODEL,
            "best_epoch": best_epoch,
            "train_loss_at_best_epoch": train_loss_best,
            "val_loss_at_best_epoch": val_loss_best,
            **summary,
        }
        append_csv_row(paths["grid"], row, GRID_RESULT_COLUMNS)

        updated = is_better_trial(summary, best_summary)
        if updated:
            best_summary = summary
            ensure_dir(paths["best_model"].parent)
            torch.save(best_state, paths["best_model"])
            save_json(
                paths["best_params"],
                {
                    "model_name": model_name,
                    "model_display": MODEL_DISPLAY[model_name],
                    "trial_id": trial_id,
                    "targets": ALL_TARGETS,
                    "bottom_targets": BOTTOM_TARGETS,
                    "mlp_hyperparameters": mlp_params,
                    "joint_batch_size": batch_size,
                    "freeze_base_branches": FREEZE_BASE_BRANCHES,
                    "initialize_from_best_model": INITIALIZE_FROM_BEST_MODEL,
                    "base_model_source_dir": str(BEST_OUTPUT_DIR),
                    "base_best_info": best_infos,
                    "best_epoch": best_epoch,
                    "train_loss_at_best_epoch": train_loss_best,
                    "val_loss_at_best_epoch": val_loss_best,
                    "best_validation_mean_nRMSE": summary["validation_mean_nRMSE"],
                    "best_validation_mean_nMAE": summary["validation_mean_nMAE"],
                    "best_validation_mean_NSE": summary["validation_mean_NSE"],
                    "best_validation_mean_KGE": summary["validation_mean_KGE"],
                    "best_validation_D1_mean_abs": summary["validation_D1_mean_abs"],
                    "best_validation_D2_mean_abs": summary["validation_D2_mean_abs"],
                    "best_validation_D1_max_abs": summary["validation_D1_max_abs"],
                    "best_validation_D2_max_abs": summary["validation_D2_max_abs"],
                    "best_validation_consistency_improvement_mean_pct": summary[
                        "validation_consistency_improvement_mean_pct"
                    ],
                    "best_validation_consistency_improvement_max_pct": summary[
                        "validation_consistency_improvement_max_pct"
                    ],
                    "best_validation_hierarchy_max_abs_error": summary[
                        "validation_hierarchy_max_abs_error"
                    ],
                },
            )
            best_metric_rows = metric_rows
            best_consistency_rows = consistency_rows
            best_consistency_summary_rows = consistency_summary_rows

        mark = " [Best Updated]" if updated else ""
        print(
            f"[{MODEL_DISPLAY[model_name]}][DynamicWLS] Trial {idx}/{len(grid)} | "
            f"MLP params={params_text(mlp_params)} | "
            f"val mean nRMSE={metric_text(summary['validation_mean_nRMSE'])} | "
            f"val D1={metric_text(summary['validation_D1_mean_abs'])} | "
            f"val D2={metric_text(summary['validation_D2_mean_abs'])} | "
            f"improve={metric_text(summary['validation_consistency_improvement_mean_pct'])}% | "
            f"constraint err={metric_text(summary['validation_hierarchy_max_abs_error'])} | "
            f"best D2={metric_text(best_summary['validation_D2_mean_abs'])} | "
            f"best nRMSE={metric_text(best_summary['validation_mean_nRMSE'])}{mark}"
        )

    if (
        best_metric_rows is None
        or best_consistency_rows is None
        or best_consistency_summary_rows is None
    ):
        (
            best_metric_rows,
            best_consistency_rows,
            best_consistency_summary_rows,
        ) = evaluate_saved_best_model(
            model_name, best_infos, prepared, batch_size, device
        )
    save_best_outputs(
        model_name,
        best_metric_rows,
        best_consistency_rows,
        best_consistency_summary_rows,
    )
    return best_metric_rows, best_consistency_rows, best_consistency_summary_rows


def evaluate_saved_best_model(
    model_name: str,
    best_infos: dict[str, dict],
    prepared: dict[str, dict],
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict], list[dict], list[dict]]:
    paths = output_paths(model_name)
    if not paths["best_params"].exists() or not paths["best_model"].exists():
        raise FileNotFoundError(f"未找到动态权重最优模型文件: {paths['best_params']}")

    best_info = json.loads(paths["best_params"].read_text(encoding="utf-8"))
    model = build_dynamic_model(
        model_name,
        best_infos,
        prepared,
        best_info["mlp_hyperparameters"],
    )
    state = torch.load(paths["best_model"], map_location="cpu")
    model.load_state_dict(state)
    return evaluate_model_on_splits(model_name, model, prepared, batch_size, device)


def accuracy_change_pct(base_value: float, reconciled_value: float, higher_is_better: bool) -> float:
    if not np.isfinite(base_value) or abs(base_value) <= EPS or not np.isfinite(reconciled_value):
        return np.nan
    if higher_is_better:
        return float((reconciled_value - base_value) / abs(base_value) * 100.0)
    return float((base_value - reconciled_value) / base_value * 100.0)


def build_final_report(
    all_metric_rows: list[dict],
    consistency_summary_rows: list[dict],
) -> pd.DataFrame:
    """专利材料表：精度变化和一致性提升分开展示，避免把协调后精度下降误解为失败。"""
    metrics_df = pd.DataFrame(all_metric_rows)
    consistency_df = pd.DataFrame(consistency_summary_rows)
    report_rows = []
    for (model, dataset, target), group in metrics_df.groupby(["model", "dataset", "target"]):
        base = group[group["method"] == "Base"]
        reconciled = group[group["method"] == "DynamicWLS"]
        if base.empty or reconciled.empty:
            continue
        base_row = base.iloc[0]
        reconciled_row = reconciled.iloc[0]
        consistency_match = consistency_df[
            (consistency_df["model"] == model) & (consistency_df["dataset"] == dataset)
        ]
        consistency_row = consistency_match.iloc[0] if not consistency_match.empty else {}
        report_rows.append(
            {
                "dataset": dataset,
                "target": target,
                "base_nRMSE": base_row["nRMSE"],
                "reconciled_nRMSE": reconciled_row["nRMSE"],
                "base_nMAE": base_row["nMAE"],
                "reconciled_nMAE": reconciled_row["nMAE"],
                "base_NSE": base_row["NSE"],
                "reconciled_NSE": reconciled_row["NSE"],
                "base_KGE": base_row["KGE"],
                "reconciled_KGE": reconciled_row["KGE"],
                "accuracy_change_nRMSE_pct": accuracy_change_pct(
                    base_row["nRMSE"], reconciled_row["nRMSE"], False
                ),
                "accuracy_change_nMAE_pct": accuracy_change_pct(
                    base_row["nMAE"], reconciled_row["nMAE"], False
                ),
                "accuracy_change_NSE_pct": accuracy_change_pct(
                    base_row["NSE"], reconciled_row["NSE"], True
                ),
                "accuracy_change_KGE_pct": accuracy_change_pct(
                    base_row["KGE"], reconciled_row["KGE"], True
                ),
                "D1_mean_abs": consistency_row.get("D1_mean_abs", np.nan),
                "D2_mean_abs": consistency_row.get("D2_mean_abs", np.nan),
                "consistency_improvement_mean_pct": consistency_row.get(
                    "consistency_improvement_mean_pct", np.nan
                ),
                "D1_max_abs": consistency_row.get("D1_max_abs", np.nan),
                "D2_max_abs": consistency_row.get("D2_max_abs", np.nan),
                "consistency_improvement_max_pct": consistency_row.get(
                    "consistency_improvement_max_pct", np.nan
                ),
            }
        )
    return pd.DataFrame(report_rows, columns=FINAL_REPORT_COLUMNS)


def save_summary(
    all_metric_rows: list[dict],
    all_consistency_summary_rows: list[dict],
) -> dict[str, Path | None]:
    summary_dir = OUTPUT_DIR / "summary"
    ensure_dir(summary_dir)
    stem = MODEL_STEM[MODEL_NAME]
    summary_path = summary_dir / f"{stem}_all_targets_dynamic_weight_metrics.csv"
    summary_df = pd.DataFrame(all_metric_rows, columns=METRIC_COLUMNS)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    summary_df.groupby(["target", "dataset", "method"], as_index=False)[
        [
            "nRMSE",
            "nMAE",
            "NSE",
            "KGE",
            "hierarchy_max_abs_error",
            "D1_mean_abs",
            "D2_mean_abs",
            "consistency_improvement_mean_pct",
        ]
    ].mean().to_csv(
        summary_dir / f"{stem}_metrics_summary_by_target.csv",
        index=False,
        encoding="utf-8-sig",
    )

    consistency_df = pd.DataFrame(
        all_consistency_summary_rows, columns=CONSISTENCY_SUMMARY_COLUMNS
    )
    consistency_summary_path = summary_dir / f"{stem}_consistency_summary.csv"
    consistency_df.groupby(["dataset"], as_index=False)[
        [
            "D1_mean_abs",
            "D2_mean_abs",
            "D1_median_abs",
            "D2_median_abs",
            "D1_max_abs",
            "D2_max_abs",
            "D1_sum_abs",
            "D2_sum_abs",
            "consistency_improvement_mean_pct",
            "consistency_improvement_median_pct",
            "consistency_improvement_max_pct",
            "consistency_improvement_sum_pct",
        ]
    ].mean().to_csv(consistency_summary_path, index=False, encoding="utf-8-sig")

    final_report_path = summary_dir / "final_report_for_patent_statement.csv"
    build_final_report(all_metric_rows, all_consistency_summary_rows).to_csv(
        final_report_path, index=False, encoding="utf-8-sig"
    )
    return {
        "summary_metrics": summary_path,
        "metrics_by_target": summary_dir / f"{stem}_metrics_summary_by_target.csv",
        "consistency_summary": consistency_summary_path,
        "final_report": final_report_path,
    }


def main() -> None:
    set_seed(SEED)
    ensure_output_dirs()
    model_name = MODEL_NAME
    if model_name != "LSTM_KAN":
        raise ValueError(f"当前脚本只允许使用 LSTM_KAN，收到: {model_name}")
    if KAN is None:
        raise ImportError("未检测到 efficient-kan，请先安装 efficient-kan 后再运行 LSTM-KAN。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df, actual_data_dir, date_col = load_data_splits(
        DATA_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV
    )
    clean_feature_cols = get_clean_feature_cols(train_df, date_col)
    print(
        f"Data: {actual_data_dir} | device={device} | "
        f"clean features={len(clean_feature_cols)} | MLP grid={len(iter_grid(MLP_GRID))} trials"
    )
    print(f"Current base model: {MODEL_DISPLAY[model_name]}")
    print(
        "Dynamic reconciliation: "
        f"initialize_from_best_model={INITIALIZE_FROM_BEST_MODEL} | "
        f"freeze_base_branches={FREEZE_BASE_BRANCHES}"
    )
    print(f"Base LSTM-KAN source: {BEST_OUTPUT_DIR}")

    print(f"\nCurrent model: {MODEL_DISPLAY[model_name]}")
    metric_rows, _, consistency_summary_rows = train_grid_for_model(
        model_name, train_df, val_df, test_df, date_col, clean_feature_cols, device
    )
    all_metric_rows = list(metric_rows)
    all_consistency_summary_rows = list(consistency_summary_rows)
    paths = output_paths(model_name)
    dynamic_val_rows = [
        row for row in metric_rows if row["dataset"] == "val" and row["method"] == "DynamicWLS"
    ]
    val_nrmse = float(np.nanmean([row["nRMSE"] for row in dynamic_val_rows]))
    val_consistency = next(row for row in consistency_summary_rows if row["dataset"] == "val")
    print(
        f"Completed model: {MODEL_DISPLAY[model_name]} | "
        f"best val mean nRMSE={metric_text(val_nrmse)} | "
        f"val D1={metric_text(val_consistency['D1_mean_abs'])} | "
        f"val D2={metric_text(val_consistency['D2_mean_abs'])} | "
        f"improve={metric_text(val_consistency['consistency_improvement_mean_pct'])}%"
    )

    summary_paths = save_summary(all_metric_rows, all_consistency_summary_rows)
    print("\nCompleted dynamic-weight end-to-end reconciliation.")
    print(f"Current base model: {MODEL_DISPLAY[model_name]}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Best dynamic-weight MLP params: {paths['best_params']}")
    print(f"Best dynamic reconciliation model: {paths['best_model']}")
    print(f"Summary metrics: {summary_paths['summary_metrics']}")
    print(f"Metrics summary by target: {summary_paths['metrics_by_target']}")
    print(f"Consistency summary: {summary_paths['consistency_summary']}")
    print(f"Patent final report: {summary_paths['final_report']}")


if __name__ == "__main__":
    main()
