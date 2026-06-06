from __future__ import annotations

import copy
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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(BASE_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_MODEL_DIR))
if str(SINGLE_OUTPUT_DIR) not in sys.path:
    sys.path.insert(0, str(SINGLE_OUTPUT_DIR))

import frozen_dynamic_wls as frozen  # noqa: E402
from base_model.common.seed_utils import set_seed  # noqa: E402
from base_model.single_output_model.grid_search_single_output import (  # noqa: E402
    DATA_DIR,
    EARLY_STOPPING_PATIENCE,
    KAN,
    MAX_EPOCHS,
    MODEL_DISPLAY,
    SEED,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
    iter_grid,
    json_safe,
    metric_text,
    params_text,
)


# =========================
# 路径与运行配置
# =========================
BEST_OUTPUT_DIR = SINGLE_OUTPUT_DIR / "grid_search_outputs"
OUTPUT_DIR = SCRIPT_DIR / "fine_tune_dynamic_wls_outputs"

MODEL_NAME = "LSTM_KAN"
MODEL_LABEL = MODEL_DISPLAY[MODEL_NAME]
FINE_TUNE_METHOD = "FineTune-DynamicWLS"
TOP_TARGET = frozen.TOP_TARGET
BOTTOM_TARGETS = frozen.BOTTOM_TARGETS
ALL_TARGETS = frozen.ALL_TARGETS
DATASETS = frozen.DATASETS

# fine-tune 版本默认继续使用 clean data 网格搜索得到的 LSTM-KAN 最优权重初始化，但不冻结基础分支。
INITIALIZE_FROM_BEST_MODEL = True
FREEZE_BASE_BRANCHES = False
JOINT_BATCH_SIZE = None
GRAD_CLIP_NORM = 5.0
EPS = 1e-12
CONSISTENCY_TOL = 1e-6
LAMBDA_CONSISTENCY = 1.0
LAMBDA_DEGRADATION = 0.5
MLP_WEIGHT_DECAY = 0.0

FINE_TUNE_GRID = {
    "base_lr": [1e-4, 5e-5],
    "base_weight_decay": [0.0, 1e-4],
    "mlp_hidden_dim": [16, 32, 64],
    "mlp_num_layers": [1, 2],
    "mlp_dropout": [0.0, 0.1],
    "mlp_lr": [1e-3, 5e-4],
    "fine_tune_scope": ["all"],
}


GRID_RESULT_COLUMNS = [
    "model_name",
    "model_display",
    "trial_id",
    "base_lr",
    "base_weight_decay",
    "mlp_hidden_dim",
    "mlp_num_layers",
    "mlp_dropout",
    "mlp_lr",
    "fine_tune_scope",
    "joint_batch_size",
    "freeze_base_branches",
    "initialize_from_best_model",
    "lambda_consistency",
    "lambda_degradation",
    "best_epoch",
    "train_total_loss_at_best_epoch",
    "val_total_loss_at_best_epoch",
    "validation_base_mean_nRMSE",
    "validation_reconciled_mean_nRMSE",
    "validation_accuracy_degradation_pct",
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

LOSS_HISTORY_COLUMNS = [
    "trial_id",
    "epoch",
    "train_total_loss",
    "train_accuracy_loss",
    "train_consistency_loss",
    "train_degradation_loss",
    "val_total_loss",
    "val_accuracy_loss",
    "val_consistency_loss",
    "val_degradation_loss",
]

OUTPUT_PATHS = {
    "grid": OUTPUT_DIR / "grid_search" / "fine_tune_dynamic_wls_grid_results.csv",
    "best_params": OUTPUT_DIR / "best_params" / "best_fine_tune_dynamic_wls.json",
    "best_model": OUTPUT_DIR / "best_models" / "best_fine_tune_dynamic_wls.pt",
    "metrics": OUTPUT_DIR / "metrics" / "fine_tune_dynamic_wls_metrics.csv",
    "consistency": OUTPUT_DIR / "consistency" / "fine_tune_dynamic_wls_consistency.csv",
    "summary": OUTPUT_DIR / "summary" / "fine_tune_dynamic_wls_final_report.csv",
    "loss_history": OUTPUT_DIR / "loss_history" / "fine_tune_dynamic_wls_loss_history.csv",
}


def ensure_output_dirs() -> None:
    for path in OUTPUT_PATHS.values():
        frozen.ensure_dir(path.parent)


def sync_best_output_source() -> None:
    # frozen_dynamic_wls 复用了路径函数；这里显式同步，避免误读旧输出目录。
    frozen.BEST_OUTPUT_DIR = BEST_OUTPUT_DIR


def require_base_grid_outputs() -> None:
    sync_best_output_source()
    if not BEST_OUTPUT_DIR.exists():
        raise FileNotFoundError(
            "未找到 clean data LSTM-KAN 网格搜索输出目录，请先完成基础模型网格搜索。"
            f"缺失路径: {BEST_OUTPUT_DIR}"
        )
    for target in ALL_TARGETS:
        params_path = frozen.best_params_path(target, MODEL_NAME)
        model_path = frozen.best_model_path(target, MODEL_NAME)
        missing = [str(path) for path in [params_path, model_path] if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"未找到 {target} 的 clean data LSTM-KAN 网格搜索最优参数或权重: {missing}"
            )


def load_base_grid_best_infos() -> dict[str, dict]:
    sync_best_output_source()
    best_infos = {target: frozen.load_best_info(target, MODEL_NAME) for target in ALL_TARGETS}
    for target, info in best_infos.items():
        source_dir = Path(info.get("base_model_source_dir", ""))
        if source_dir.resolve() != BEST_OUTPUT_DIR.resolve():
            raise ValueError(
                f"{target} 的最佳超参数来源不是 clean data 网格搜索输出目录: {source_dir}"
            )
    return best_infos


def mlp_params_for_trial(params: dict) -> dict:
    return {
        "hidden_dim": params["mlp_hidden_dim"],
        "num_layers": params["mlp_num_layers"],
        "dropout": params["mlp_dropout"],
        "lr": params["mlp_lr"],
    }


def configure_fine_tune_scope(
    model: frozen.DynamicHierarchicalReconciliationModel,
    scope: str,
) -> None:
    """
    控制基础 LSTM-KAN 分支的可训练范围。
    当前搜索空间默认使用 all；保留 kan_head 和 last_lstm_layer 是为了后续可以做更保守的局部微调。
    """
    valid_scopes = {"kan_head", "last_lstm_layer", "all"}
    if scope not in valid_scopes:
        raise ValueError(f"未知 fine_tune_scope={scope}，可选范围为 {sorted(valid_scopes)}")

    for branch in model.branches.values():
        for param in branch.parameters():
            param.requires_grad = False

        if scope == "all":
            for param in branch.parameters():
                param.requires_grad = True
        elif scope == "kan_head":
            for name, param in branch.named_parameters():
                if name.startswith("kan."):
                    param.requires_grad = True
        elif scope == "last_lstm_layer":
            num_layers = getattr(branch.lstm, "num_layers", None)
            if num_layers is None:
                raise AttributeError("LSTM-KAN 分支缺少 lstm.num_layers，无法定位最后一层 LSTM。")
            last_suffix = f"_l{num_layers - 1}"
            for name, param in branch.named_parameters():
                if name.startswith("lstm.") and last_suffix in name:
                    param.requires_grad = True

    for param in model.weight_mlp.parameters():
        param.requires_grad = True

    trainable_branch_params = [
        param
        for branch in model.branches.values()
        for param in branch.parameters()
        if param.requires_grad
    ]
    if not trainable_branch_params:
        raise ValueError(f"fine_tune_scope={scope} 没有选中任何基础 LSTM-KAN 分支参数。")


def build_fine_tune_model(
    best_infos: dict[str, dict],
    prepared: dict[str, dict],
    trial_params: dict,
) -> frozen.DynamicHierarchicalReconciliationModel:
    old_init = frozen.INITIALIZE_FROM_BEST_MODEL
    old_freeze = frozen.FREEZE_BASE_BRANCHES
    try:
        frozen.INITIALIZE_FROM_BEST_MODEL = INITIALIZE_FROM_BEST_MODEL
        frozen.FREEZE_BASE_BRANCHES = FREEZE_BASE_BRANCHES
        model = frozen.build_dynamic_model(
            MODEL_NAME,
            best_infos,
            prepared,
            mlp_params_for_trial(trial_params),
        )
    finally:
        frozen.INITIALIZE_FROM_BEST_MODEL = old_init
        frozen.FREEZE_BASE_BRANCHES = old_freeze

    configure_fine_tune_scope(model, trial_params["fine_tune_scope"])
    return model


def optimizer_for_trial(
    model: frozen.DynamicHierarchicalReconciliationModel,
    trial_params: dict,
) -> torch.optim.Optimizer:
    base_params = [
        param
        for branch in model.branches.values()
        for param in branch.parameters()
        if param.requires_grad
    ]
    mlp_params = [param for param in model.weight_mlp.parameters() if param.requires_grad]
    if not base_params:
        raise ValueError("联合微调需要至少一个可训练的基础 LSTM-KAN 参数。")
    if not mlp_params:
        raise ValueError("联合微调需要可训练的动态权重 MLP 参数。")

    # 基础分支和协调 MLP 的学习率尺度不同，分组优化能避免已训练好的 LSTM-KAN 被过大步长破坏。
    return torch.optim.AdamW(
        [
            {
                "params": base_params,
                "lr": trial_params["base_lr"],
                "weight_decay": trial_params["base_weight_decay"],
            },
            {
                "params": mlp_params,
                "lr": trial_params["mlp_lr"],
                "weight_decay": MLP_WEIGHT_DECAY,
            },
        ]
    )


def scaled_log_mse(
    values: torch.Tensor,
    y_true: torch.Tensor,
    model: frozen.DynamicHierarchicalReconciliationModel,
) -> torch.Tensor:
    """
    在 log1p 后的标准化尺度计算 MSE，使不同藻类目标的量纲更接近。
    预测值可能因微调出现小幅负值，因此只裁剪到 log1p 的有效下界附近。
    """
    safe_values = torch.clamp(values, min=-0.999999)
    safe_true = torch.clamp(y_true, min=-0.999999)
    value_log = torch.log1p(safe_values)
    true_log = torch.log1p(safe_true)
    means = model.target_means.unsqueeze(0)
    scales = torch.clamp(model.target_scales.unsqueeze(0), min=EPS)
    value_scaled = (value_log - means) / scales
    true_scaled = (true_log - means) / scales
    return torch.mean((value_scaled - true_scaled) ** 2)


def loss_components(
    output: dict[str, torch.Tensor],
    y_true: torch.Tensor,
    model: frozen.DynamicHierarchicalReconciliationModel,
) -> dict[str, torch.Tensor]:
    accuracy_loss = scaled_log_mse(output["reconciled"], y_true, model)
    base_mse = scaled_log_mse(output["base_pred"], y_true, model)
    consistency_residual = output["reconciled"][:, 0] - output["reconciled"][:, 1:].sum(dim=1)
    consistency_loss = torch.mean(torch.abs(consistency_residual))

    # 退化惩罚只把当前基础预测当作参照，不让模型通过刻意拉差 base 预测来降低该项。
    degradation_loss = torch.relu(accuracy_loss - base_mse.detach())
    total_loss = (
        accuracy_loss
        + LAMBDA_CONSISTENCY * consistency_loss
        + LAMBDA_DEGRADATION * degradation_loss
    )
    return {
        "total": total_loss,
        "accuracy": accuracy_loss,
        "consistency": consistency_loss,
        "degradation": degradation_loss,
    }


def run_epoch(
    model: frozen.DynamicHierarchicalReconciliationModel,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = {"total": 0.0, "accuracy": 0.0, "consistency": 0.0, "degradation": 0.0}
    n_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            branch_inputs = [item.to(device) for item in batch[:-1]]
            y_true = batch[-1].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(branch_inputs)
            losses = loss_components(output, y_true, model)
            if training:
                losses["total"].backward()
                if GRAD_CLIP_NORM is not None:
                    trainable_params = [p for p in model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP_NORM)
                optimizer.step()

            batch_n = y_true.size(0)
            n_samples += batch_n
            for key in sums:
                sums[key] += float(losses[key].detach().cpu()) * batch_n

    return {key: value / max(n_samples, 1) for key, value in sums.items()}


def train_fine_tune_model(
    model: frozen.DynamicHierarchicalReconciliationModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[list[dict], dict, int, float, float]:
    model.to(device)
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    train_loss_at_best = float("inf")
    wait = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_losses = run_epoch(model, train_loader, device, optimizer)
        val_losses = run_epoch(model, val_loader, device, None)
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": train_losses["total"],
                "train_accuracy_loss": train_losses["accuracy"],
                "train_consistency_loss": train_losses["consistency"],
                "train_degradation_loss": train_losses["degradation"],
                "val_total_loss": val_losses["total"],
                "val_accuracy_loss": val_losses["accuracy"],
                "val_consistency_loss": val_losses["consistency"],
                "val_degradation_loss": val_losses["degradation"],
            }
        )

        if val_losses["total"] < best_val_loss - 1e-12:
            best_val_loss = val_losses["total"]
            best_epoch = epoch
            train_loss_at_best = train_losses["total"]
            best_state = frozen._state_dict_to_cpu(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is None:
        best_state = frozen._state_dict_to_cpu(model.state_dict())
    model.load_state_dict(copy.deepcopy(best_state))
    return history, best_state, best_epoch, train_loss_at_best, best_val_loss


def metric_rows_for_split(
    dataset: str,
    predictions: dict[str, np.ndarray],
    consistency_summary: dict,
) -> list[dict]:
    base_residual = frozen.hierarchy_residual(predictions["base"])
    reconciled_residual = frozen.hierarchy_residual(predictions["reconciled"])
    rows = []
    for target_idx, target in enumerate(ALL_TARGETS):
        for method, key, residual in [
            ("Base", "base", base_residual),
            (FINE_TUNE_METHOD, "reconciled", reconciled_residual),
        ]:
            metrics = frozen.compute_required_metrics(
                predictions["true"][:, target_idx],
                predictions[key][:, target_idx],
            )
            rows.append(
                {
                    "model": MODEL_LABEL,
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


def evaluate_model_on_splits(
    model: frozen.DynamicHierarchicalReconciliationModel,
    prepared: dict[str, dict],
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict], list[dict], list[dict]]:
    metric_rows = []
    consistency_rows = []
    consistency_summary_rows = []
    for dataset in DATASETS:
        loader = frozen.make_joint_loader(prepared, dataset, batch_size, False, seed=SEED)
        predictions = frozen.predict_dynamic_model(model, loader, device)
        split_consistency_rows = frozen.consistency_rows_for_split(
            MODEL_NAME, dataset, frozen.split_dates(prepared, dataset), predictions
        )
        split_consistency_summary = frozen.summarize_consistency(split_consistency_rows)
        consistency_rows.extend(split_consistency_rows)
        consistency_summary_rows.append(split_consistency_summary)
        metric_rows.extend(metric_rows_for_split(dataset, predictions, split_consistency_summary))
    return metric_rows, consistency_rows, consistency_summary_rows


def accuracy_degradation_pct(base_value: float, reconciled_value: float) -> float:
    if not np.isfinite(base_value) or abs(base_value) <= EPS or not np.isfinite(reconciled_value):
        return np.nan
    return float((reconciled_value - base_value) / base_value * 100.0)


def validation_summary(rows: list[dict], consistency_summaries: list[dict]) -> dict:
    base_rows = [row for row in rows if row["dataset"] == "val" and row["method"] == "Base"]
    reconciled_rows = [
        row for row in rows if row["dataset"] == "val" and row["method"] == FINE_TUNE_METHOD
    ]
    if not base_rows or not reconciled_rows:
        raise ValueError("验证集指标为空，无法选择 fine-tune trial。")
    val_consistency = next(row for row in consistency_summaries if row["dataset"] == "val")

    base_mean_nrmse = float(np.nanmean([row["nRMSE"] for row in base_rows]))
    reconciled_mean_nrmse = float(np.nanmean([row["nRMSE"] for row in reconciled_rows]))
    return {
        "validation_base_mean_nRMSE": base_mean_nrmse,
        "validation_reconciled_mean_nRMSE": reconciled_mean_nrmse,
        "validation_accuracy_degradation_pct": accuracy_degradation_pct(
            base_mean_nrmse, reconciled_mean_nrmse
        ),
        "validation_mean_nRMSE": reconciled_mean_nrmse,
        "validation_mean_nMAE": float(np.nanmean([row["nMAE"] for row in reconciled_rows])),
        "validation_mean_NSE": float(np.nanmean([row["NSE"] for row in reconciled_rows])),
        "validation_mean_KGE": float(np.nanmean([row["KGE"] for row in reconciled_rows])),
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
            np.nanmax([row["hierarchy_max_abs_error"] for row in reconciled_rows])
        ),
    }


def is_better_trial(current_summary: dict, best_summary: dict | None) -> bool:
    """
    选择 trial 时先保证层次一致性；当 D2 已接近 0 时，再用协调后平均 nRMSE 比较精度。
    这样避免只追求一致性而牺牲预测性能。
    """
    if best_summary is None:
        return True

    current_d2 = float(current_summary.get("validation_D2_mean_abs", float("inf")))
    best_d2 = float(best_summary.get("validation_D2_mean_abs", float("inf")))
    current_nrmse = float(
        current_summary.get("validation_reconciled_mean_nRMSE", float("inf"))
    )
    best_nrmse = float(best_summary.get("validation_reconciled_mean_nRMSE", float("inf")))

    current_consistent = np.isfinite(current_d2) and current_d2 <= CONSISTENCY_TOL
    best_consistent = np.isfinite(best_d2) and best_d2 <= CONSISTENCY_TOL
    if current_consistent and not best_consistent:
        return True
    if current_consistent and best_consistent:
        return current_nrmse < best_nrmse
    if not current_consistent and not best_consistent:
        if abs(current_d2 - best_d2) <= CONSISTENCY_TOL:
            return current_nrmse < best_nrmse
        return current_d2 < best_d2
    return False


def build_final_report(
    metric_rows: list[dict],
    consistency_summary_rows: list[dict],
) -> pd.DataFrame:
    metrics_df = pd.DataFrame(metric_rows)
    consistency_df = pd.DataFrame(consistency_summary_rows)
    report_rows = []
    for (dataset, target), group in metrics_df.groupby(["dataset", "target"]):
        base = group[group["method"] == "Base"]
        reconciled = group[group["method"] == FINE_TUNE_METHOD]
        if base.empty or reconciled.empty:
            continue
        base_row = base.iloc[0]
        reconciled_row = reconciled.iloc[0]
        consistency_match = consistency_df[consistency_df["dataset"] == dataset]
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
                "accuracy_change_nRMSE_pct": frozen.accuracy_change_pct(
                    base_row["nRMSE"], reconciled_row["nRMSE"], False
                ),
                "accuracy_change_nMAE_pct": frozen.accuracy_change_pct(
                    base_row["nMAE"], reconciled_row["nMAE"], False
                ),
                "accuracy_change_NSE_pct": frozen.accuracy_change_pct(
                    base_row["NSE"], reconciled_row["NSE"], True
                ),
                "accuracy_change_KGE_pct": frozen.accuracy_change_pct(
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
    return pd.DataFrame(report_rows, columns=frozen.FINAL_REPORT_COLUMNS)


def save_grid_results(rows: list[dict]) -> None:
    pd.DataFrame(rows, columns=GRID_RESULT_COLUMNS).to_csv(
        OUTPUT_PATHS["grid"], index=False, encoding="utf-8-sig"
    )


def save_loss_history_rows(rows: list[dict]) -> None:
    pd.DataFrame(rows, columns=LOSS_HISTORY_COLUMNS).to_csv(
        OUTPUT_PATHS["loss_history"], index=False, encoding="utf-8-sig"
    )


def save_best_artifacts(
    best_state: dict,
    best_payload: dict,
    metric_rows: list[dict],
    consistency_rows: list[dict],
    consistency_summary_rows: list[dict],
) -> None:
    torch.save(best_state, OUTPUT_PATHS["best_model"])
    frozen.save_json(OUTPUT_PATHS["best_params"], best_payload)
    pd.DataFrame(metric_rows, columns=frozen.METRIC_COLUMNS).to_csv(
        OUTPUT_PATHS["metrics"], index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(consistency_rows, columns=frozen.CONSISTENCY_COLUMNS).to_csv(
        OUTPUT_PATHS["consistency"], index=False, encoding="utf-8-sig"
    )
    build_final_report(metric_rows, consistency_summary_rows).to_csv(
        OUTPUT_PATHS["summary"], index=False, encoding="utf-8-sig"
    )


def train_grid(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str | None,
    clean_feature_cols: list[str],
    device: torch.device,
) -> tuple[dict, list[dict], list[dict]]:
    best_infos = load_base_grid_best_infos()
    prepared = frozen.prepare_all_target_data(
        best_infos, train_df, val_df, test_df, date_col, clean_feature_cols
    )
    batch_size = JOINT_BATCH_SIZE or frozen.joint_batch_size(best_infos)

    grid = iter_grid(FINE_TUNE_GRID)
    grid_rows = []
    loss_history_rows = []
    best_summary = None
    best_state = None
    best_payload = None
    best_metric_rows = None
    best_consistency_rows = None
    best_consistency_summary_rows = None

    for idx, trial_params in enumerate(grid, start=1):
        trial_id = f"trial_{idx:04d}"
        set_seed(SEED)
        model = build_fine_tune_model(best_infos, prepared, trial_params)
        train_loader = frozen.make_joint_loader(prepared, "train", batch_size, True, seed=SEED)
        val_loader = frozen.make_joint_loader(prepared, "val", batch_size, False, seed=SEED)
        optimizer = optimizer_for_trial(model, trial_params)

        history, trial_state, best_epoch, train_loss_best, val_loss_best = train_fine_tune_model(
            model, train_loader, val_loader, optimizer, device
        )
        for row in history:
            loss_history_rows.append({"trial_id": trial_id, **row})

        metric_rows, consistency_rows, consistency_summary_rows = evaluate_model_on_splits(
            model, prepared, batch_size, device
        )
        summary = validation_summary(metric_rows, consistency_summary_rows)
        grid_row = {
            "model_name": MODEL_NAME,
            "model_display": MODEL_LABEL,
            "trial_id": trial_id,
            **trial_params,
            "joint_batch_size": batch_size,
            "freeze_base_branches": FREEZE_BASE_BRANCHES,
            "initialize_from_best_model": INITIALIZE_FROM_BEST_MODEL,
            "lambda_consistency": LAMBDA_CONSISTENCY,
            "lambda_degradation": LAMBDA_DEGRADATION,
            "best_epoch": best_epoch,
            "train_total_loss_at_best_epoch": train_loss_best,
            "val_total_loss_at_best_epoch": val_loss_best,
            **summary,
        }
        grid_rows.append(grid_row)
        save_grid_results(grid_rows)
        save_loss_history_rows(loss_history_rows)

        updated = is_better_trial(summary, best_summary)
        if updated:
            best_summary = summary
            best_state = trial_state
            best_metric_rows = metric_rows
            best_consistency_rows = consistency_rows
            best_consistency_summary_rows = consistency_summary_rows
            best_payload = {
                "model_name": MODEL_NAME,
                "model_display": MODEL_LABEL,
                "trial_id": trial_id,
                "targets": ALL_TARGETS,
                "bottom_targets": BOTTOM_TARGETS,
                "fine_tune_hyperparameters": trial_params,
                "mlp_hyperparameters": mlp_params_for_trial(trial_params),
                "joint_batch_size": batch_size,
                "fine_tune_scope": trial_params["fine_tune_scope"],
                "freeze_base_branches": FREEZE_BASE_BRANCHES,
                "initialize_from_best_model": INITIALIZE_FROM_BEST_MODEL,
                "base_model_source_dir": str(BEST_OUTPUT_DIR),
                "base_best_info": best_infos,
                "lambda_consistency": LAMBDA_CONSISTENCY,
                "lambda_degradation": LAMBDA_DEGRADATION,
                "best_epoch": best_epoch,
                "train_total_loss_at_best_epoch": train_loss_best,
                "val_total_loss_at_best_epoch": val_loss_best,
                **{f"best_{key}": value for key, value in summary.items()},
            }
            save_best_artifacts(
                best_state,
                best_payload,
                best_metric_rows,
                best_consistency_rows,
                best_consistency_summary_rows,
            )

        mark = " [Best Updated]" if updated else ""
        print(
            f"[{MODEL_LABEL}][FineTune-DynamicWLS] Trial {idx}/{len(grid)} | "
            f"params={params_text(trial_params)} | "
            f"val base nRMSE={metric_text(summary['validation_base_mean_nRMSE'])} | "
            f"val reconciled nRMSE={metric_text(summary['validation_reconciled_mean_nRMSE'])} | "
            f"degradation={metric_text(summary['validation_accuracy_degradation_pct'])}% | "
            f"val D1={metric_text(summary['validation_D1_mean_abs'])} | "
            f"val D2={metric_text(summary['validation_D2_mean_abs'])} | "
            f"improve={metric_text(summary['validation_consistency_improvement_mean_pct'])}%{mark}"
        )

    if (
        best_payload is None
        or best_state is None
        or best_metric_rows is None
        or best_consistency_rows is None
        or best_consistency_summary_rows is None
    ):
        raise RuntimeError("fine-tune grid 未产生有效 trial，请检查数据、权重和搜索空间。")

    save_best_artifacts(
        best_state,
        best_payload,
        best_metric_rows,
        best_consistency_rows,
        best_consistency_summary_rows,
    )
    return best_payload, best_metric_rows, best_consistency_summary_rows


def print_final_summary(
    best_payload: dict,
    metric_rows: list[dict],
    consistency_summary_rows: list[dict],
) -> None:
    val_base_rows = [
        row for row in metric_rows if row["dataset"] == "val" and row["method"] == "Base"
    ]
    val_reconciled_rows = [
        row
        for row in metric_rows
        if row["dataset"] == "val" and row["method"] == FINE_TUNE_METHOD
    ]
    val_consistency = next(row for row in consistency_summary_rows if row["dataset"] == "val")
    val_base_nrmse = float(np.nanmean([row["nRMSE"] for row in val_base_rows]))
    val_reconciled_nrmse = float(np.nanmean([row["nRMSE"] for row in val_reconciled_rows]))

    print("\nCompleted fine-tune dynamic WLS reconciliation.")
    print(f"Best trial params: {json.dumps(json_safe(best_payload['fine_tune_hyperparameters']), ensure_ascii=False)}")
    print(
        "Validation mean nRMSE: "
        f"Base={metric_text(val_base_nrmse)} | "
        f"{FINE_TUNE_METHOD}={metric_text(val_reconciled_nrmse)}"
    )
    print(
        "Validation consistency: "
        f"D1_mean_abs={metric_text(val_consistency['D1_mean_abs'])} | "
        f"D2_mean_abs={metric_text(val_consistency['D2_mean_abs'])}"
    )
    print(
        "Validation consistency improvement: "
        f"{metric_text(val_consistency['consistency_improvement_mean_pct'])}%"
    )
    print(f"Final outputs saved to: {OUTPUT_DIR}")


def main() -> None:
    set_seed(SEED)
    ensure_output_dirs()
    require_base_grid_outputs()
    if MODEL_NAME != "LSTM_KAN":
        raise ValueError(f"当前脚本只支持 LSTM_KAN，收到: {MODEL_NAME}")
    if KAN is None:
        raise ImportError("未检测到 efficient-kan，请先安装 efficient-kan 后再运行 LSTM-KAN。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df, actual_data_dir, date_col = frozen.load_data_splits(
        DATA_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV
    )
    clean_feature_cols = frozen.get_clean_feature_cols(train_df, date_col)
    print(
        f"Data: {actual_data_dir} | device={device} | "
        f"clean features={len(clean_feature_cols)} | "
        f"fine-tune grid={len(iter_grid(FINE_TUNE_GRID))} trials"
    )
    print(f"Current base model: {MODEL_LABEL}")
    print(
        "Fine-tune dynamic reconciliation: "
        f"initialize_from_best_model={INITIALIZE_FROM_BEST_MODEL} | "
        f"freeze_base_branches={FREEZE_BASE_BRANCHES}"
    )
    print(f"Base LSTM-KAN source: {BEST_OUTPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    best_payload, metric_rows, consistency_summary_rows = train_grid(
        train_df, val_df, test_df, date_col, clean_feature_cols, device
    )
    print_final_summary(best_payload, metric_rows, consistency_summary_rows)


if __name__ == "__main__":
    main()
