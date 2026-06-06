#!/usr/bin/env Rscript

# LSTM-KAN SHAP + PDP 综合图绘制脚本。
# 本脚本只读取已有 SHAP 输出和 clean data 网格搜索后的 LSTM-KAN 最优权重，不修改训练或 SHAP 分析结果。
required_packages <- c(
  "ggplot2",
  "dplyr",
  "readr",
  "patchwork",
  "scales",
  "tibble",
  "reticulate"
)

missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop(
    "Missing required R packages: ",
    paste(missing_packages, collapse = ", "),
    ". Please install them before running this script.",
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(patchwork)
  library(scales)
  library(tibble)
})

# =========================
# 字体与路径设置
# =========================
# 图中可见文字保持英文；代码注释使用中文，便于后续自行微调图片。
font_family <- "Arial"
if (.Platform$OS.type == "windows") {
  # Windows 下显式注册 Arial，减少导出 PDF/PNG 时字体回退的概率。
  grDevices::windowsFonts(Arial = grDevices::windowsFont("Arial"))
}

get_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  file_idx <- grep(file_arg, args, fixed = TRUE)
  if (length(file_idx) > 0) {
    return(normalizePath(sub(file_arg, "", args[file_idx[1]], fixed = TRUE), winslash = "/", mustWork = FALSE))
  }
  normalizePath("visualization/shap_plots/lstm_kan_shap_plots.R", winslash = "/", mustWork = FALSE)
}

script_path <- get_script_path()
project_root <- normalizePath(file.path(dirname(script_path), "..", ".."), winslash = "/", mustWork = TRUE)

shap_dir <- file.path(project_root, "SHAP", "shap_lstm_kan_outputs")
output_dir <- file.path(project_root, "visualization", "shap_plots", "plots")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

targets_to_run <- c(
  "Green_Algae",
  "Cyanobacteria",
  "Diatoms",
  "Cryptophyta",
  "Algae_Sum"
)

# 目标变量显示标签：只影响图中英文显示，不改变读取数据时使用的原始 target 名。
target_labels <- c(
  Green_Algae = "Green algae",
  Cyanobacteria = "Cyanobacteria",
  Diatoms = "Diatoms",
  Cryptophyta = "Cryptophyta",
  Algae_Sum = "Total algae"
)

# =========================
# 可调绘图参数
# =========================
# SHAP 重要性柱状图设置。
importance_top_n <- 15             # 每个目标展示的前 N 个重要特征。
importance_bar_width <- 0.72       # 柱体厚度；越大柱子越粗。
importance_bar_linewidth <- 0.18   # 柱体边框线宽；过大容易显得沉重。
importance_grid_color <- "gray88"  # SHAP 重要性图 x 方向辅助网格线颜色。
importance_grid_linewidth <- 0.45  # SHAP 重要性图 x 方向辅助网格线线宽。
importance_fill_color <- "#6F8CB8" # 单一特征重要性柱体填充色；不再区分环境/藻类因子。
importance_line_color <- "#426C9B" # 柱体边框颜色，与项目内 LSTM-MLP 深蓝线色保持协调。

# PDP 小图设置。
pdp_feature_count <- 4             # 每个目标展示前几个重要特征的 PDP。
pdp_lag_index <- 1                 # 缺少 lag-level SHAP 时的备用 lag；正常情况下会使用 SHAP 最大的 lag。
pdp_grid_size <- 40                # PDP 横轴采样点数；越大曲线越平滑但计算更慢。
pdp_max_samples <- 300             # PDP 使用的最大测试样本数；越大越稳定但计算更慢。
pdp_random_seed <- 42              # PDP 抽样随机种子，保证重复运行结果一致。
pdp_line_color <- "#426C9B"        # PDP 曲线颜色。
pdp_linewidth <- 0.80              # PDP 曲线线宽。
pdp_point_size <- 1.5             # PDP 散点大小；用于显示曲线采样位置，避免喧宾夺主。
pdp_point_alpha <- 0.28            # PDP 散点透明度；数值越低，散点越淡。

# 主题与导出设置。
base_text_size <- 15               # 全图字号固定为 15，包含标题框、坐标轴、刻度和图例。
panel_border_linewidth <- 1.00     # 每个子图黑色外边框线宽。
major_grid_linewidth <- 0.45       # 横向/纵向主网格线线宽。
strip_fill <- "grey80"             # 子图标题框填充色；修改这里会影响所有灰色标题框。
legend_key_height_cm <- 0.35       # 图例色块高度；当前脚本无图例，保留以维持主题设置一致。
legend_key_width_cm <- 0.65        # 图例色块宽度；当前脚本无图例，保留以维持主题设置一致。
output_width <- 14.5               # 导出图像宽度，单位为英寸。
output_height <- 7.8               # 导出图像高度，单位为英寸。
output_dpi <- 300                  # PNG 导出分辨率；论文图可改为 600。

base_theme <- function() {
  # 统一主题说明：
  # - 背景保持白色，便于论文和 PPT 直接使用；
  # - 每个面板保留黑色外边框，增强多子图结构感；
  # - 子图标题使用灰色 strip 框，便于在组合图中快速区分面板；
  # - 次网格线关闭，主网格线在具体图层中按需要开启；
  # - 图例默认放在底部，避免遮挡柱状图和 PDP 曲线。
  theme_bw(base_family = font_family, base_size = base_text_size) +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      panel.border = element_rect(fill = NA, color = "black", linewidth = panel_border_linewidth),
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = strip_fill, color = "black", linewidth = panel_border_linewidth),
      strip.text = element_text(face = "bold", hjust = 0.5, size = base_text_size, color = "black"),
      plot.title = element_text(face = "bold", hjust = 0.5, size = base_text_size),
      axis.title = element_text(size = base_text_size),
      axis.text = element_text(size = base_text_size, color = "black"),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = base_text_size),
      legend.key.height = grid::unit(legend_key_height_cm, "cm"),
      legend.key.width = grid::unit(legend_key_width_cm, "cm"),
      plot.margin = margin(6, 6, 6, 6)
    )
}

target_label <- function(x) {
  x <- as.character(x)
  label <- target_labels[x]
  label[is.na(label)] <- feature_label_base(x[is.na(label)])
  unname(label)
}

feature_label_base <- function(x) {
  # 只修改图中显示名称，不改变原始数据列名或模型输入列名。
  label <- as.character(x)
  label <- gsub("^log_", "", label, ignore.case = TRUE)
  label <- gsub("_", " ", label, fixed = TRUE)
  label <- gsub("\\s+", " ", trimws(label))

  label[label %in% c("Green Algae", "Green algae")] <- "Green algae"
  label[label %in% c("Algae Sum", "Alage Sum", "Algae sum", "Alage sum")] <- "Total algae"
  label[label %in% c("precipitation", "Precipitation")] <- "Precipitation"
  label
}

feature_label <- function(x, lag_index = NULL) {
  label <- feature_label_base(x)
  if (!is.null(lag_index)) {
    label <- paste0(label, " (lag ", lag_index, ")")
  }
  label
}

# =========================
# SHAP 重要性读取与绘图
# =========================
read_shap_importance <- function() {
  feature_path <- file.path(shap_dir, "shap_feature_importance_no_self.csv")
  if (!file.exists(feature_path)) {
    stop("Cannot find SHAP feature importance file: ", feature_path, call. = FALSE)
  }

  feature_importance <- read_csv(feature_path, show_col_types = FALSE)
  required_cols <- c("target", "feature", "mean_abs_shap", "rank")
  missing_cols <- setdiff(required_cols, names(feature_importance))
  if (length(missing_cols) > 0) {
    stop(
      "SHAP feature importance file is missing required columns: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }

  feature_importance %>%
    mutate(
      target = as.character(.data$target),
      feature = as.character(.data$feature),
      mean_abs_shap = as.numeric(.data$mean_abs_shap),
      rank = as.integer(.data$rank)
    )
}

read_shap_best_lags <- function() {
  best_lag_path <- file.path(shap_dir, "shap_feature_best_lag_no_self.csv")
  if (!file.exists(best_lag_path)) {
    stop("Cannot find SHAP best-lag file: ", best_lag_path, call. = FALSE)
  }

  best_lags <- read_csv(best_lag_path, show_col_types = FALSE)
  required_cols <- c("target", "feature", "best_lag", "best_lag_time_index", "best_lag_mean_abs_shap")
  missing_cols <- setdiff(required_cols, names(best_lags))
  if (length(missing_cols) > 0) {
    stop(
      "SHAP best-lag file is missing required columns: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }

  best_lags %>%
    mutate(
      target = as.character(.data$target),
      feature = as.character(.data$feature),
      best_lag = as.integer(.data$best_lag),
      best_lag_time_index = as.integer(.data$best_lag_time_index),
      best_lag_mean_abs_shap = as.numeric(.data$best_lag_mean_abs_shap)
    )
}

attach_best_lags <- function(target_importance, best_lags, target_name) {
  # 每个 target-feature 使用 SHAP 在 30 个 lag 中最大的 lag 来命名和绘制 PDP。
  target_lags <- best_lags %>%
    filter(.data$target == target_name) %>%
    select(all_of(c("target", "feature", "best_lag", "best_lag_time_index", "best_lag_mean_abs_shap")))

  merged <- target_importance %>%
    left_join(target_lags, by = c("target", "feature"))

  missing_lags <- merged %>%
    filter(is.na(.data$best_lag)) %>%
    pull("feature")
  if (length(missing_lags) > 0) {
    stop(
      "Cannot find best SHAP lag for target ",
      target_name,
      ": ",
      paste(missing_lags, collapse = ", "),
      call. = FALSE
    )
  }

  merged
}

clean_target_importance <- function(feature_importance, target_name) {
  # no-self SHAP 文件已去掉目标自身历史项；这里不再按环境/藻类分组，所有特征放在同一排序中比较。
  target_importance <- feature_importance %>%
    filter(.data$target == target_name, !is.na(.data$mean_abs_shap)) %>%
    group_by(.data$target, .data$feature) %>%
    summarise(mean_abs_shap = max(.data$mean_abs_shap, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(.data$mean_abs_shap), .data$feature) %>%
    mutate(
      rank = row_number(),
      normalized_importance = .data$mean_abs_shap / sum(.data$mean_abs_shap, na.rm = TRUE)
    )

  target_importance
}

plot_shap_importance <- function(feature_importance, best_lags, target_name, top_n = importance_top_n) {
  panel_title <- paste0("(a) SHAP feature importance for ", target_label(target_name))
  importance <- clean_target_importance(feature_importance, target_name) %>%
    attach_best_lags(best_lags, target_name) %>%
    slice_head(n = top_n) %>%
    mutate(
      feature_display = feature_label_base(.data$feature),
      feature_display = factor(.data$feature_display, levels = rev(.data$feature_display)),
      panel_title = panel_title
    )

  if (nrow(importance) == 0) {
    stop("No SHAP feature importance rows for target: ", target_name, call. = FALSE)
  }

  ggplot(importance, aes(x = .data$mean_abs_shap, y = .data$feature_display)) +
    geom_col(
      width = importance_bar_width,
      fill = importance_fill_color,
      color = importance_line_color,
      linewidth = importance_bar_linewidth
    ) +
    scale_x_continuous(
      labels = label_number(accuracy = 0.001),
      expand = expansion(mult = c(0, 0.06))
    ) +
    labs(
      x = "Mean |SHAP value|",
      y = NULL
    ) +
    facet_wrap(~panel_title) +
    base_theme() +
    theme(
      panel.grid.major.x = element_line(color = importance_grid_color, linewidth = importance_grid_linewidth),
      panel.grid.major.y = element_blank(),
      legend.position = "none"
    )
}

# =========================
# Python PDP 计算
# =========================
choose_python_for_reticulate <- function() {
  # reticulate 有时会自动绑定到一个没有 pandas/torch 的 Python。
  # 这里优先使用用户显式指定的 RETICULATE_PYTHON，其次使用当前 conda 环境，最后回退到 PATH。
  conda_prefix <- Sys.getenv("CONDA_PREFIX", unset = "")
  conda_python <- if (nzchar(conda_prefix)) {
    if (.Platform$OS.type == "windows") {
      file.path(conda_prefix, "python.exe")
    } else {
      file.path(conda_prefix, "bin", "python")
    }
  } else {
    ""
  }

  candidates <- c(
    Sys.getenv("RETICULATE_PYTHON", unset = ""),
    conda_python,
    unname(Sys.which("python")),
    unname(Sys.which("python3"))
  )
  candidates <- unique(candidates[nzchar(candidates)])
  candidates <- candidates[file.exists(candidates)]

  if (length(candidates) == 0) {
    return(NA_character_)
  }
  normalizePath(candidates[[1]], winslash = "/", mustWork = FALSE)
}

configure_reticulate_python <- function() {
  if (!reticulate::py_available(initialize = FALSE)) {
    python_path <- choose_python_for_reticulate()
    if (is.na(python_path)) {
      stop(
        "No Python executable found for reticulate. Set RETICULATE_PYTHON to the project Python environment.",
        call. = FALSE
      )
    }
    message("Using Python for PDP via reticulate: ", python_path)
    reticulate::use_python(python_path, required = TRUE)
  } else {
    message("reticulate Python is already initialized; using existing Python session.")
  }

  config <- reticulate::py_config()
  message("reticulate Python: ", config$python)

  reticulate::py_run_string(
    "
import importlib.util
required_modules = ['numpy', 'pandas', 'torch', 'sklearn', 'efficient_kan']
missing_python_modules = [
    module for module in required_modules
    if importlib.util.find_spec(module) is None
]
"
  )
  missing_modules <- unlist(reticulate::py$missing_python_modules)
  if (length(missing_modules) > 0) {
    stop(
      "Selected Python is missing required modules: ",
      paste(missing_modules, collapse = ", "),
      ". Install them in this Python or set RETICULATE_PYTHON to the environment used by the project.",
      call. = FALSE
    )
  }

  TRUE
}

init_python_pdp <- function() {
  configure_reticulate_python()

  # Python 侧直接复用项目的 clean data 工具函数和 LSTM-KAN 构建函数，避免在 R 脚本中重复实现模型结构。
  py_code <- sprintf(
    '
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(r"""%s""")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base_model.common.data_utils import (
    get_clean_feature_cols,
    inverse_log_target,
    load_data_splits,
    prepare_single_target_data,
)
from base_model.single_output_model.grid_search_single_output import HORIZON, build_model

DATA_DIR = PROJECT_ROOT / "base_model" / "data"
BEST_OUTPUT_DIR = (
    PROJECT_ROOT
    / "base_model"
    / "single_output_model"
    / "grid_search_outputs"
)
TARGET_COLS = {
    "Green_Algae": "log_Green_Algae",
    "Cyanobacteria": "log_Cyanobacteria",
    "Diatoms": "log_Diatoms",
    "Cryptophyta": "log_Cryptophyta",
    "Algae_Sum": "log_Algae_Sum",
}

_bundle_cache = {}

def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")

def _load_best_info(target: str) -> dict:
    params_path = BEST_OUTPUT_DIR / "best_params" / target / "best_params_lstm_kan.json"
    model_path = BEST_OUTPUT_DIR / "best_models" / target / "best_lstm_kan.pt"
    _require_file(params_path, "LSTM-KAN clean data grid-search best parameter file")
    _require_file(model_path, "LSTM-KAN clean data grid-search best model file")

    with params_path.open("r", encoding="utf-8") as f:
        best_info = json.load(f)
    params = best_info.get("used_hyperparameters") or best_info.get("best_hyperparameters")
    if params is None:
        raise KeyError(f"Cannot find used_hyperparameters or best_hyperparameters in {params_path}.")

    feature_cols = best_info.get("feature_cols")
    if not feature_cols:
        raise KeyError(f"Cannot find feature_cols in {params_path}.")

    return {
        "params_path": params_path,
        "model_path": model_path,
        "params": params,
        "feature_cols": list(feature_cols),
        "target_col": best_info.get("target_col", TARGET_COLS[target]),
    }

def load_target_bundle(target: str) -> dict:
    if target in _bundle_cache:
        return _bundle_cache[target]
    if target not in TARGET_COLS:
        raise ValueError(f"Unknown target: {target}")

    best_info = _load_best_info(target)
    train_df, val_df, test_df, _, date_col = load_data_splits(DATA_DIR)

    feature_cols = list(best_info["feature_cols"])
    clean_feature_cols = get_clean_feature_cols(train_df, date_col)
    missing_from_clean = [col for col in feature_cols if col not in clean_feature_cols]
    if missing_from_clean:
        raise ValueError(
            "Best parameter feature_cols contain columns absent from current clean data columns. "
            f"target={target}, missing_from_clean={missing_from_clean}, "
            f"params={best_info[\'params_path\']}"
        )
    if feature_cols != clean_feature_cols:
        extra_clean = [col for col in clean_feature_cols if col not in feature_cols]
        print(
            "WARNING: Best parameter feature_cols do not exactly match current "
            "get_clean_feature_cols columns. Using saved feature_cols order to match model weights. "
            f"target={target}, extra_clean={extra_clean}",
            file=sys.stderr,
        )

    params = dict(best_info["params"])
    lookback = int(params.get("lookback", 30))
    data = prepare_single_target_data(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=best_info["target_col"],
        lookback=lookback,
        horizon=HORIZON,
        date_col=date_col,
    )

    model = build_model("LSTM_KAN", len(feature_cols), params)
    state = torch.load(best_info["model_path"], map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    bundle = {
        "params_path": str(best_info["params_path"]),
        "model_path": str(best_info["model_path"]),
        "feature_cols": feature_cols,
        "data": data,
        "model": model,
    }
    _bundle_cache[target] = bundle
    return bundle

def compute_pdp_for_target_feature(target, feature, lag=1, grid_size=40, max_samples=300, random_seed=42):
    bundle = load_target_bundle(target)
    feature_cols = bundle["feature_cols"]
    if feature not in feature_cols:
        raise ValueError(f"Feature {feature} is not in clean-feature model inputs for {target}.")

    feature_idx = feature_cols.index(feature)
    data = bundle["data"]
    model = bundle["model"]
    x_scaler = data["x_scaler"]
    y_scaler = data["y_scaler"]
    X_test = data["test"]["X"].copy()

    rng = np.random.default_rng(int(random_seed))
    if len(X_test) > int(max_samples):
        sample_idx = np.sort(rng.choice(len(X_test), size=int(max_samples), replace=False))
        X_base = X_test[sample_idx].copy()
    else:
        X_base = X_test.copy()

    lag = int(lag)
    lookback = X_base.shape[1]
    if lag < 1 or lag > lookback:
        raise ValueError(f"lag must be in [1, {lookback}], got {lag} for {target} / {feature}.")
    time_index = lookback - lag

    lag_values_scaled = X_base[:, time_index, feature_idx]
    low, high = np.nanpercentile(lag_values_scaled, [5, 95])
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"Non-finite PDP grid bounds for {target} / {feature} lag {lag}.")
    if abs(high - low) < 1e-12:
        low, high = np.nanmin(lag_values_scaled), np.nanmax(lag_values_scaled)
    if abs(high - low) < 1e-12:
        high = low + 1e-6

    grid_scaled = np.linspace(low, high, int(grid_size))
    mean_predictions = []
    for value in grid_scaled:
        X_mod = X_base.copy()
        # PDP 只替换 SHAP 最大的那个 lag，其它时间步和其它特征保持不变。
        X_mod[:, time_index, feature_idx] = value
        with torch.no_grad():
            pred_scaled = model(torch.from_numpy(X_mod).float()).detach().cpu().numpy()
        pred_original = inverse_log_target(pred_scaled, y_scaler)
        mean_predictions.append(float(np.mean(pred_original)))

    feature_mean = float(x_scaler.mean_[feature_idx])
    feature_scale = float(x_scaler.scale_[feature_idx])
    grid_display = grid_scaled * feature_scale + feature_mean

    return pd.DataFrame({
        "target": target,
        "feature": feature,
        "feature_value": grid_display,
        "feature_value_scaled": grid_scaled,
        "mean_prediction": mean_predictions,
        "lag": lag,
        "time_index": time_index,
        "n_samples": len(X_base),
        "params_path": bundle["params_path"],
        "model_path": bundle["model_path"],
    })
',
    project_root
  )

  reticulate::py_run_string(py_code)
  TRUE
}

compute_pdp_for_feature <- function(
    target_name,
    feature,
    rank,
    lag_index,
    grid_size = pdp_grid_size,
    max_samples = pdp_max_samples,
    random_seed = pdp_random_seed) {
  pdp_data <- reticulate::py$compute_pdp_for_target_feature(
    target_name,
    feature,
    as.integer(lag_index),
    as.integer(grid_size),
    as.integer(max_samples),
    as.integer(random_seed)
  )
  pdp_data <- as_tibble(pdp_data)
  pdp_data$rank <- rank
  pdp_data$lag <- as.integer(lag_index)
  pdp_data
}

plot_single_pdp <- function(pdp_data, feature, lag_index, panel_label) {
  # PDP 面板：
  # - 横轴是 SHAP 最大 lag 的特征原始尺度取值；
  # - 纵轴是模型在该特征取值下的平均预测；
  # - 低透明度散点显示 PDP 采样位置，曲线仍作为主要趋势表达。
  display_feature <- feature_label(feature, lag_index = lag_index)
  plot_data <- pdp_data %>%
    mutate(panel_title = paste0(panel_label, " ", display_feature))

  ggplot(plot_data, aes(x = .data$feature_value, y = .data$mean_prediction)) +
    geom_point(color = pdp_line_color, size = pdp_point_size, alpha = pdp_point_alpha) +
    geom_line(color = pdp_line_color, linewidth = pdp_linewidth) +
    labs(
      x = display_feature,
      y = "Mean prediction"
    ) +
    facet_wrap(~panel_title) +
    base_theme() +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(color = "gray88", linewidth = major_grid_linewidth),
      legend.position = "none"
    )
}

assemble_target_plot <- function(importance_plot, pdp_plots, target_name) {
  # 拼图结构：
  # - 左侧为 SHAP 重要性排序图；
  # - 右侧为前 pdp_feature_count 个重要特征的 PDP 小图，默认 2 x 2 排列；
  # - 不再输出环境因子/藻类因子分组图，减少文件数量并保持解释口径一致。
  pdp_panel <- wrap_plots(pdp_plots, ncol = 2)
  importance_plot + pdp_panel +
    plot_layout(widths = c(1, 1.35), guides = "collect") &
    theme(
      legend.position = "bottom",
      plot.background = element_rect(fill = "white", color = NA)
    )
}

save_target_plot <- function(plot_obj, target_name) {
  png_path <- file.path(output_dir, paste0(target_name, "_lstm_kan_shap_pdp.png"))
  pdf_path <- file.path(output_dir, paste0(target_name, "_lstm_kan_shap_pdp.pdf"))

  # 导出设置：
  # - PNG 用于快速查看和插入文档；
  # - PDF 用于论文或矢量编辑；
  # - bg = "white" 避免透明背景在 Word/PPT 中变黑。
  ggsave(png_path, plot = plot_obj, width = output_width, height = output_height, dpi = output_dpi, bg = "white")
  tryCatch(
    ggsave(pdf_path, plot = plot_obj, width = output_width, height = output_height, device = grDevices::cairo_pdf, bg = "white"),
    error = function(e) {
      message("cairo_pdf failed, falling back to default PDF device: ", conditionMessage(e))
      ggsave(pdf_path, plot = plot_obj, width = output_width, height = output_height, device = "pdf", bg = "white")
    }
  )
  message("Saved: ", png_path)
  message("Saved: ", pdf_path)
}

feature_importance <- read_shap_importance()
shap_best_lags <- read_shap_best_lags()
init_python_pdp()

for (target_name in targets_to_run) {
  message("\nProcessing target: ", target_name)

  target_importance <- clean_target_importance(feature_importance, target_name) %>%
    attach_best_lags(shap_best_lags, target_name)
  if (nrow(target_importance) == 0) {
    message("No SHAP feature importance rows for target: ", target_name, ". Skipping.")
    next
  }

  importance_plot <- plot_shap_importance(
    feature_importance,
    shap_best_lags,
    target_name,
    top_n = importance_top_n
  )
  top_features <- target_importance %>%
    slice_head(n = min(pdp_feature_count, nrow(target_importance))) %>%
    select(all_of(c("feature", "rank", "best_lag", "best_lag_time_index")))

  pdp_plots <- vector("list", nrow(top_features))
  for (i in seq_len(nrow(top_features))) {
    feature <- top_features$feature[[i]]
    rank <- top_features$rank[[i]]
    best_lag <- top_features$best_lag[[i]]

    message("  PDP rank ", rank, ": ", feature, " at lag ", best_lag)
    pdp_data <- compute_pdp_for_feature(
      target_name,
      feature,
      rank,
      best_lag,
      grid_size = pdp_grid_size,
      max_samples = pdp_max_samples,
      random_seed = pdp_random_seed
    )
    pdp_plots[[i]] <- plot_single_pdp(pdp_data, feature, best_lag, paste0("(", letters[i + 1], ")"))
  }

  combined_plot <- assemble_target_plot(importance_plot, pdp_plots, target_name)
  save_target_plot(combined_plot, target_name)
}

message("\nAll LSTM-KAN SHAP/PDP plots have been processed.")
