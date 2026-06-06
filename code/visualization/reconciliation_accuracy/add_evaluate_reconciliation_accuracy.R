#!/usr/bin/env Rscript

# Reconciliation accuracy figure
# 图中显示文字保持英文；中文注释用于说明关键路径、指标计算和绘图设置，便于后续修改美化。

required_packages <- c("ggplot2", "dplyr", "tidyr", "scales")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
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
  library(tidyr)
  library(scales)
})

EPS <- 1e-12

# =========================
# 可调参数
# =========================
# 只绘制测试集结果；如需看验证集或训练集，可改为 "val" 或 "train"。
dataset_to_plot <- "test"

# 静态协调方法来自 hierarchical_reconciliation，DynamicWLS 来自 clean-feature 端到端协调输出。
standard_model_to_plot <- "LSTM-KAN"
dynamic_model_label <- "LSTM-KAN"

# x 轴目标顺序和图中英文标签；只改变显示，不改变数据读取时的原始 target 名。
target_order <- c(
  "Green_Algae",
  "Cyanobacteria",
  "Diatoms",
  "Cryptophyta",
  "Algae_Sum"
)
target_labels <- c(
  Green_Algae = "Green algae",
  Cyanobacteria = "Cyanobacteria",
  Diatoms = "Diatoms",
  Cryptophyta = "Cryptophyta",
  Algae_Sum = "Total algae"
)

# 图例顺序；若要隐藏某个方法，可从这里删掉。
# ★ 新增 MinT，位于 BU 之后、OLS 之前，与静态协调方法排列逻辑一致。
method_order <- c("Base", "BU", "MinT", "OLS", "WLS", "DynamicWLS")

# 柱子填充色：在原有五色基础上，为 MinT 插入一个介于 BU (#b4deb6) 和 OLS (#7bc6be) 之间的
# 过渡色 #98d4bc，保持整体由浅绿→深蓝的渐变视觉逻辑不变。
method_fill_colors <- c(
  Base       = "#eaf3e2",
  BU         = "#b4deb6",
  MinT       = "#98d4bc",   # ★ 新增
  OLS        = "#7bc6be",
  WLS        = "#439cc4",
  DynamicWLS = "#0868a6"
)

# 柱子边框色：与填充色对应，MinT 边框色取 BU 和 OLS 边框色的中间色调。
method_line_colors <- c(
  Base       = "#B7C9AC",
  BU         = "#72B978",
  MinT       = "#55AC98",   # ★ 新增
  OLS        = "#3B9F98",
  WLS        = "#2179A0",
  DynamicWLS = "#064A78"
)

# 四个子图顺序与 multi_seed_boxplots.R 保持一致。
metric_order <- c("NSE", "KGE", "nMAE", "nRMSE")
metric_titles <- c(
  NSE    = "(a) NSE",
  KGE    = "(b) KGE",
  nMAE   = "(c) nMAE",
  nRMSE  = "(d) nRMSE"
)

# 柱状图和主题参数集中放在这里，后续微调图形时优先修改这些变量。
target_group_spacing <- 1.60   # 不同藻类组中心之间的距离，越大不同藻类之间留白越明显。
method_bar_spacing   <- 0.235  # 同一藻类内相邻方法柱子的中心间距，应略大于 bar_width。
bar_width            <- 0.22   # 单个柱子的宽度；接近 method_bar_spacing 时，同一藻类柱子会挨在一起。
bar_linewidth        <- 0.65   # 柱子边框线宽。
panel_border_linewidth <- 1.00 # 每个子图黑色外边框线宽。
grid_linewidth       <- 0.45   # 浅灰横向网格线线宽。
strip_fill           <- "grey80" # 子图标题条背景色。
base_font_size       <- 15     # 全图基础字号。
figure_width         <- 12.8   # 输出图宽度，单位为英寸。
figure_height        <- 8.2    # 输出图高度，单位为英寸。
figure_dpi           <- 600    # PNG 分辨率；PDF 为矢量格式，不受该参数影响。
figure_name <- "add_reconciliation_accuracy_metrics" # 输出文件名主干；脚本会同时保存 PNG 和 PDF。
use_cairo_pdf <- TRUE          # PDF 使用 Cairo 设备，避免 Windows 基础 pdf 设备漏绘 Arial 文字。
y_axis_lower <- 0              # 固定纵坐标下限为 0，使零点位于面板底端并贴近 x 轴。

font_family <- "Arial"
if (.Platform$OS.type == "windows") {
  grDevices::windowsFonts(Arial = grDevices::windowsFont("Arial"))
}

script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- sub("^--file=", "", file_arg[1])
    return(dirname(normalizePath(script_path, winslash = "/", mustWork = TRUE)))
  }
  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

find_project_root <- function(start_dir) {
  current <- normalizePath(start_dir, winslash = "/", mustWork = TRUE)
  while (TRUE) {
    has_standard <- dir.exists(file.path(current, "hierarchical_reconciliation"))
    has_dynamic  <- dir.exists(file.path(current, "dynamic_weight_end_to_end_reconciliation"))
    if (has_standard && has_dynamic) {
      return(current)
    }
    parent <- dirname(current)
    if (identical(parent, current)) {
      stop(
        "Could not find project root containing hierarchical_reconciliation and ",
        "dynamic_weight_end_to_end_reconciliation.",
        call. = FALSE
      )
    }
    current <- parent
  }
}

read_csv_checked <- function(path) {
  if (!file.exists(path)) {
    stop("Missing required file: ", path, call. = FALSE)
  }
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

first_existing_column <- function(data, candidates, file_label) {
  matched <- candidates[candidates %in% names(data)]
  if (length(matched) == 0) {
    stop(
      "Missing required column in ", file_label, ". Expected one of: ",
      paste(candidates, collapse = ", "),
      call. = FALSE
    )
  }
  matched[[1]]
}

valid_pair <- function(observed, predicted) {
  obs  <- as.numeric(observed)
  pred <- as.numeric(predicted)
  keep <- is.finite(obs) & is.finite(pred)
  list(obs = obs[keep], pred = pred[keep])
}

rmse <- function(obs, pred) {
  if (length(obs) == 0) return(NA_real_)
  sqrt(mean((pred - obs)^2))
}

mae <- function(obs, pred) {
  if (length(obs) == 0) return(NA_real_)
  mean(abs(pred - obs))
}

nse <- function(obs, pred) {
  if (length(obs) == 0) return(NA_real_)
  denominator <- sum((obs - mean(obs))^2)
  if (abs(denominator) < EPS) return(NA_real_)
  1 - sum((pred - obs)^2) / denominator
}

kge <- function(obs, pred) {
  if (length(obs) < 2) return(NA_real_)
  obs_std  <- stats::sd(obs)
  pred_std <- stats::sd(pred)
  obs_mean  <- mean(obs)
  pred_mean <- mean(pred)
  if (obs_std < EPS || pred_std < EPS || abs(obs_mean) < EPS) return(NA_real_)
  corr <- suppressWarnings(stats::cor(obs, pred))
  if (!is.finite(corr)) return(NA_real_)
  alpha <- pred_std / obs_std
  beta  <- pred_mean / obs_mean
  1 - sqrt((corr - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
}

metric_row <- function(target, method, observed, predicted) {
  pair       <- valid_pair(observed, predicted)
  group_rmse <- rmse(pair$obs, pair$pred)
  group_mae  <- mae(pair$obs, pair$pred)
  obs_mean   <- if (length(pair$obs) == 0) NA_real_ else mean(pair$obs)
  data.frame(
    target = target,
    method = method,
    KGE    = kge(pair$obs, pair$pred),
    NSE    = nse(pair$obs, pair$pred),
    nRMSE  = ifelse(is.finite(obs_mean) && abs(obs_mean) >= EPS, group_rmse / obs_mean, NA_real_),
    nMAE   = ifelse(is.finite(obs_mean) && abs(obs_mean) >= EPS, group_mae  / obs_mean, NA_real_),
    stringsAsFactors = FALSE
  )
}

read_standard_metrics <- function(project_root) {
  prediction_root <- file.path(
    project_root,
    "hierarchical_reconciliation",
    "reconciliation_outputs"
  )
  if (!dir.exists(prediction_root)) {
    stop("Missing static reconciliation output directory: ", prediction_root, call. = FALSE)
  }

  prediction_files <- list.files(
    prediction_root,
    pattern    = "_reconciled_predictions\\.csv$",
    recursive  = TRUE,
    full.names = TRUE
  )
  prediction_files <- prediction_files[!grepl("[/\\\\]summary[/\\\\]", prediction_files)]
  if (length(prediction_files) == 0) {
    stop("No static reconciliation prediction files were found in: ", prediction_root, call. = FALSE)
  }

  # ★ 新增 MinT = "MinT_pred"：与 BU/OLS/WLS 采用相同读取逻辑，
  #   列不存在时静默跳过（next），不会因单个文件缺列而中止整体运行。
  prediction_columns <- c(
    Base = "base_pred",
    BU   = "BU_pred",
    MinT = "MinT_pred",   # ★ 新增
    OLS  = "OLS_pred",
    WLS  = "WLS_pred"
  )

  rows <- list()
  for (path in prediction_files) {
    data <- read_csv_checked(path)
    if ("model" %in% names(data)) {
      data <- data[data$model == standard_model_to_plot, , drop = FALSE]
    }
    if (nrow(data) == 0) next

    target_col   <- first_existing_column(data, c("target", "series", "variable"), path)
    observed_col <- first_existing_column(data, c("y_true", "observed", "actual", "truth"), path)

    for (method in names(prediction_columns)) {
      pred_col <- prediction_columns[[method]]
      if (!pred_col %in% names(data)) next
      grouped <- split(data, data[[target_col]])
      rows <- c(
        rows,
        lapply(names(grouped), function(target) {
          metric_row(
            target    = target,
            method    = method,
            observed  = grouped[[target]][[observed_col]],
            predicted = grouped[[target]][[pred_col]]
          )
        })
      )
    }
  }

  if (length(rows) == 0) {
    stop(
      "No static reconciliation rows were available for model: ",
      standard_model_to_plot,
      call. = FALSE
    )
  }

  bind_rows(rows) %>%
    group_by(.data$target, .data$method) %>%
    summarise(
      across(all_of(metric_order), ~ mean(.x, na.rm = TRUE)),
      .groups = "drop"
    )
}

read_dynamic_wls_metrics <- function(project_root) {
  dynamic_metrics_path <- file.path(
    project_root,
    "dynamic_weight_end_to_end_reconciliation",
    "frozen_dynamic_wls_outputs",
    "summary",
    "lstm_kan_all_targets_dynamic_weight_metrics.csv"
  )
  data <- read_csv_checked(dynamic_metrics_path)
  required_cols <- c("model", "target", "dataset", "method", metric_order)
  missing_cols  <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0) {
    stop(
      "DynamicWLS metrics file is missing columns: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }

  dynamic_rows <- data %>%
    filter(
      .data$model   == dynamic_model_label,
      tolower(.data$dataset) == dataset_to_plot,
      .data$method  == "DynamicWLS"
    ) %>%
    select(target, method, all_of(metric_order))

  if (nrow(dynamic_rows) == 0) {
    stop(
      "No DynamicWLS rows were found for dataset='",
      dataset_to_plot,
      "' in: ",
      dynamic_metrics_path,
      call. = FALSE
    )
  }

  dynamic_rows
}

prepare_plot_data <- function(metrics) {
  metrics %>%
    filter(.data$target %in% target_order, .data$method %in% method_order) %>%
    mutate(
      target = factor(.data$target, levels = target_order, labels = target_labels[target_order]),
      method = factor(.data$method, levels = method_order)
    ) %>%
    pivot_longer(
      cols      = all_of(metric_order),
      names_to  = "metric",
      values_to = "value"
    ) %>%
    mutate(
      metric        = factor(.data$metric, levels = metric_order, labels = metric_titles[metric_order]),
      target_index  = as.numeric(.data$target),
      method_index  = as.numeric(.data$method),
      x_position    = (target_index - 1) * target_group_spacing +
        (method_index - (length(method_order) + 1) / 2) * method_bar_spacing
    )
}

make_accuracy_plot <- function(plot_data) {
  # 手动数值 x 坐标用于同时控制两层间距：同一藻类内部柱子贴近，不同藻类之间保留留白。
  target_axis_positions <- (seq_along(target_order) - 1) * target_group_spacing
  target_axis_labels    <- unname(target_labels[target_order])

  # 只在 KGE 和 NSE 面板绘制 y = 0 参考线。
  zero_line_data <- data.frame(
    metric = factor(c(metric_titles["KGE"], metric_titles["NSE"]), levels = metric_titles[metric_order]),
    x      = min(plot_data$x_position, na.rm = TRUE) - bar_width / 2,
    xend   = max(plot_data$x_position, na.rm = TRUE) + bar_width / 2,
    y      = 0,
    yend   = 0
  )

  ggplot(plot_data) +
    geom_segment(
      data = zero_line_data,
      aes(x = .data$x, xend = .data$xend, y = .data$y, yend = .data$yend),
      color     = "gray45",
      linewidth = 0.45
    ) +
    geom_col(
      aes(x = .data$x_position, y = .data$value, fill = .data$method, color = .data$method),
      width     = bar_width,
      linewidth = bar_linewidth,
      na.rm     = TRUE
    ) +
    scale_fill_manual(values  = method_fill_colors, drop = FALSE) +
    scale_color_manual(values = method_line_colors, drop = FALSE, guide = "none") +
    scale_x_continuous(
      breaks = target_axis_positions,
      labels = target_axis_labels,
      expand = expansion(mult = c(0.035, 0.035))
    ) +
    scale_y_continuous(
      limits = c(y_axis_lower, NA),
      breaks = function(limits) sort(unique(c(y_axis_lower, scales::breaks_pretty(n = 4)(limits)))),
      labels = scales::label_number(accuracy = 0.01),
      expand = expansion(mult = c(0.005, 0.12))
    ) +
    facet_wrap(~ metric, ncol = 2, scales = "free_y") +
    labs(x = NULL, y = NULL, fill = NULL) +
    theme_bw(base_family = font_family, base_size = base_font_size) +
    theme(
      plot.background    = element_rect(fill = "white", color = NA),
      panel.background   = element_rect(fill = "white", color = NA),
      panel.border       = element_rect(fill = NA, color = "black", linewidth = panel_border_linewidth),
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      panel.grid.major.y = element_line(color = "gray88", linewidth = grid_linewidth),
      panel.grid.minor.y = element_blank(),
      strip.background   = element_rect(fill = strip_fill, color = "black", linewidth = panel_border_linewidth),
      strip.text         = element_text(size = base_font_size, hjust = 0.5, margin = margin(t = 6, b = 6)),
      axis.text.x        = element_text(size = base_font_size, color = "black", angle = 25, hjust = 1),
      axis.text.y        = element_text(size = base_font_size, color = "black"),
      axis.title.y       = element_text(size = base_font_size),
      legend.position    = "bottom",
      legend.direction   = "horizontal",
      legend.background  = element_blank(),
      legend.box.background = element_blank(),
      legend.key         = element_rect(fill = "white", color = NA),
      legend.text        = element_text(size = base_font_size),
      panel.spacing      = grid::unit(1.0, "lines"),
      plot.margin        = margin(8, 10, 8, 10)
    ) +
    guides(fill = guide_legend(nrow = 1, byrow = TRUE))
}

main <- function() {
  root       <- find_project_root(script_dir())
  output_dir <- file.path(root, "visualization", "reconciliation_accuracy", "plots")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  standard_metrics <- read_standard_metrics(root)
  dynamic_metrics  <- read_dynamic_wls_metrics(root)
  metrics          <- bind_rows(standard_metrics, dynamic_metrics)
  plot_data        <- prepare_plot_data(metrics)

  if (nrow(plot_data) == 0) {
    stop("No metric rows are available for plotting.", call. = FALSE)
  }

  accuracy_plot <- make_accuracy_plot(plot_data)

  # 同一张图同时导出 PNG 和 PDF。
  pdf_output_path <- file.path(output_dir, paste0(figure_name, ".pdf"))
  png_output_path <- file.path(output_dir, paste0(figure_name, ".png"))

  if (use_cairo_pdf && !capabilities("cairo")) {
    stop(
      "Cairo graphics support is required for complete PDF text output. ",
      "Please enable Cairo in R or set use_cairo_pdf <- FALSE and use a PDF-safe font family.",
      call. = FALSE
    )
  }

  ggsave(
    pdf_output_path,
    plot   = accuracy_plot,
    width  = figure_width,
    height = figure_height,
    device = if (use_cairo_pdf) grDevices::cairo_pdf else "pdf",
    bg     = "white",
    family = font_family
  )
  ggsave(
    png_output_path,
    plot   = accuracy_plot,
    width  = figure_width,
    height = figure_height,
    dpi    = figure_dpi,
    bg     = "white"
  )

  message("Saved PDF: ", normalizePath(pdf_output_path, winslash = "/", mustWork = TRUE))
  message("Saved PNG: ", normalizePath(png_output_path, winslash = "/", mustWork = TRUE))
}

main()
