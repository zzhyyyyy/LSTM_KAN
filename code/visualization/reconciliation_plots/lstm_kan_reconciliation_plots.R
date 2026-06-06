if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop(
    paste(
      "Package 'ggplot2' is required.",
      "Install it with install.packages('ggplot2')."
    )
  )
}
if (!requireNamespace("scales", quietly = TRUE)) {
  stop(
    paste(
      "Package 'scales' is required.",
      "Install it with install.packages('scales')."
    )
  )
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

invisible(Sys.setlocale("LC_TIME", "C"))

font_family <- "Arial"
if (.Platform$OS.type == "windows") {
  grDevices::windowsFonts(Arial = grDevices::windowsFont("Arial"))
}

# =========================
# 绘图参数设置
# =========================
# 这些参数集中控制字号、边框、图例和导出尺寸，后续美化时优先修改这一段。
base_font_size <- 15
panel_border_linewidth <- 1.00
strip_border_linewidth <- 0.75
plot_width <- 11.8
plot_height <- 12.8
plot_dpi <- 600

# 只绘制以 LSTM-KAN 为基础模型的层次协调预测结果。
base_model_to_plot <- "LSTM-KAN"

# 五个藻类目标在多面板图中的自上而下顺序。
target_order <- c(
  "Green_Algae",
  "Cyanobacteria",
  "Diatoms",
  "Cryptophyta",
  "Algae_Sum"
)
target_labels <- c(
  Green_Algae = "(a) Green algae",
  Cyanobacteria = "(b) Cyanobacteria",
  Diatoms = "(c) Diatoms",
  Cryptophyta = "(d) Cryptophyta",
  Algae_Sum = "(e) Total algae"
)

# reconciliation_outputs 中各协调方法对应的预测列。
method_order <- c("Base", "BU", "TD", "OLS", "WLS", "MinT", "DynamicWLS")
prediction_columns <- c(
  "base_pred",
  "BU_pred",
  "TD_pred",
  "OLS_pred",
  "WLS_pred",
  "MinT_pred",
  "DynamicWLS_pred"
)
prediction_labels <- c(
  base_pred = "Base",
  BU_pred = "BU",
  TD_pred = "TD",
  OLS_pred = "OLS",
  WLS_pred = "WLS",
  MinT_pred = "MinT",
  DynamicWLS_pred = "DynamicWLS"
)

# 曲线颜色和线型。Observed 用黑色实线，各协调方法用不同颜色区分。
series_levels <- c("Observed", method_order)
series_colors <- c(
  Observed = "#222222",
  Base = "#8C8C8C",
  BU = "#1b3d69",
  TD = "#F2A93B",
  OLS = "#457B9D",
  WLS = "#74C476",
  MinT = "#7B3294",
  DynamicWLS = "#e34141"
)
series_linetypes <- c(
  Observed = "solid",
  Base = "dashed",
  BU = "dashed",
  TD = "dashed",
  OLS = "dashed",
  WLS = "dashed",
  MinT = "dotdash",
  DynamicWLS = "solid"
)
series_linewidths <- c(
  Observed = 0.86,
  Base = 0.62,
  BU = 0.62,
  TD = 0.62,
  OLS = 0.62,
  WLS = 0.62,
  MinT = 0.68,
  DynamicWLS = 0.78
)

date_breaks <- scales::breaks_width("1 month")
date_labels <- scales::label_date("%Y-%m")
value_breaks <- scales::breaks_pretty(n = 4)
value_labels <- scales::label_number(big.mark = ",", trim = TRUE)
y_axis_label <- expression("Algal biomass (" * mu * "g/L)")

script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- sub("^--file=", "", file_arg[1])
    return(dirname(normalizePath(script_path, winslash = "/", mustWork = TRUE)))
  }
  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

project_root <- function(start_dir) {
  current <- start_dir
  marker <- file.path(
    "hierarchical_reconciliation",
    "reconciliation_outputs",
    "Green_Algae",
    "Green_Algae_reconciled_predictions.csv"
  )

  while (!file.exists(file.path(current, marker))) {
    parent <- dirname(current)
    if (identical(parent, current)) {
      stop("Could not find project root containing: ", marker)
    }
    current <- parent
  }
  current
}

read_dynamic_target_predictions <- function(dynamic_reconciliation_root, target_name) {
  path <- file.path(
    dynamic_reconciliation_root,
    "predictions",
    "lstm_kan_frozen_dynamic_wls_predictions.csv"
  )
  if (!file.exists(path)) {
    stop("Missing dynamic-weight prediction file: ", path)
  }

  data <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  required_cols <- c("date", "sample_index", "model", "target", "DynamicWLS_pred")
  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0) {
    stop(
      "DynamicWLS prediction file is missing columns: ",
      paste(missing_cols, collapse = ", ")
    )
  }
  data <- data[data$model == base_model_to_plot & data$target == target_name, ]
  if (nrow(data) == 0) {
    stop(
      "No rows found for ",
      base_model_to_plot,
      " and target ",
      target_name,
      " in dynamic-weight file: ",
      path
    )
  }
  data$date <- as.Date(data$date)
  data$DynamicWLS_pred <- as.numeric(data$DynamicWLS_pred)
  data[
    ,
    c("date", "sample_index", "target", "DynamicWLS_pred")
  ]
}

read_target_predictions <- function(reconciliation_root, dynamic_reconciliation_root, target_name) {
  path <- file.path(
    reconciliation_root,
    target_name,
    paste0(target_name, "_reconciled_predictions.csv")
  )
  if (!file.exists(path)) {
    stop("Missing prediction file: ", path)
  }

  data <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  if ("date" %in% names(data) && any(!is.na(data$date))) {
    data$date <- as.Date(data$date)
  }
  data <- data[data$model == base_model_to_plot, ]
  if (nrow(data) == 0) {
    stop("No rows found for model ", base_model_to_plot, " in: ", path)
  }
  dynamic_data <- read_dynamic_target_predictions(dynamic_reconciliation_root, target_name)
  data <- merge(
    data,
    dynamic_data,
    by = c("date", "sample_index", "target"),
    all.x = TRUE,
    sort = FALSE
  )
  if (any(is.na(data$DynamicWLS_pred))) {
    stop("DynamicWLS predictions could not be aligned for target: ", target_name)
  }
  data
}

prediction_long <- function(data) {
  observed <- unique(
    data[, c("date", "sample_index", "target", "y_true")]
  )
  observed$series <- "Observed"
  observed$value <- observed$y_true

  predictions <- do.call(
    rbind,
    lapply(prediction_columns, function(column_name) {
      if (!column_name %in% names(data)) {
        stop("Missing prediction column: ", column_name)
      }
      out <- data[, c("date", "sample_index", "target")]
      out$series <- prediction_labels[[column_name]]
      out$value <- data[[column_name]]
      out
    })
  )

  combined <- rbind(
    observed[, c("date", "sample_index", "target", "series", "value")],
    predictions[, c("date", "sample_index", "target", "series", "value")]
  )
  combined$series <- factor(combined$series, levels = series_levels)
  combined$target_panel <- factor(
    target_labels[combined$target],
    levels = target_labels[target_order]
  )
  if ("date" %in% names(combined) && any(!is.na(combined$date))) {
    combined$x_value <- combined$date
  } else {
    combined$x_value <- combined$sample_index
  }
  combined
}

make_panel_plot <- function(plot_data) {
  ggplot(
    plot_data,
    aes(
      x = .data$x_value,
      y = .data$value,
      color = .data$series,
      linetype = .data$series,
      linewidth = .data$series,
      group = interaction(.data$target_panel, .data$series)
    )
  ) +
    geom_line(na.rm = TRUE, alpha = 0.94) +
    facet_wrap(vars(.data$target_panel), ncol = 1, scales = "free_y") +
    scale_color_manual(values = series_colors, drop = FALSE) +
    scale_linetype_manual(values = series_linetypes, drop = FALSE) +
    scale_linewidth_manual(values = series_linewidths, drop = FALSE) +
    scale_y_continuous(
      breaks = value_breaks,
      labels = value_labels,
      expand = expansion(mult = c(0.08, 0.14))
    ) +
    scale_x_date(
      breaks = date_breaks,
      labels = date_labels,
      expand = expansion(mult = c(0.005, 0.01))
    ) +
    labs(x = NULL, y = y_axis_label) +
    guides(
      color = guide_legend(
        nrow = 2,
        byrow = TRUE,
        override.aes = list(
          linetype = unname(series_linetypes),
          linewidth = unname(series_linewidths)
        )
      ),
      linetype = "none",
      linewidth = "none"
    ) +
    theme_bw(base_size = base_font_size, base_family = font_family) +
    theme(
      # 白色背景用于论文和汇报排版，避免导出后出现透明背景。
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      # 每个子图保留完整黑色外框，线宽统一为 1.00，增强多面板结构感。
      panel.border = element_rect(fill = NA, color = "black", linewidth = panel_border_linewidth),
      # 时间序列图只保留浅灰横向主网格，帮助读取纵坐标且减少纵向视觉干扰。
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      panel.grid.major.y = element_line(color = "gray88", linewidth = 0.45),
      panel.grid.minor.y = element_blank(),
      panel.spacing = grid::unit(0.62, "lines"),
      # 灰色标题框明确绑定子图标题和对应面板，便于论文中引用子图编号。
      strip.background = element_rect(
        fill = "grey80",
        color = "black",
        linewidth = strip_border_linewidth
      ),
      strip.text = element_text(size = base_font_size, hjust = 0.5, face = "bold"),
      axis.title.y = element_text(size = base_font_size, margin = margin(r = 9)),
      axis.title.x = element_blank(),
      axis.text.x = element_text(size = base_font_size, color = "black", angle = 0),
      axis.text.y = element_text(size = base_font_size, color = "black"),
      axis.ticks = element_line(color = "black", linewidth = 0.35),
      # 图例移到底部并保留边框；颜色映射沿用原脚本，不在此处改变。
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.justification = "center",
      legend.background = element_rect(fill = "white", color = "black", linewidth = 0.40),
      legend.box.background = element_blank(),
      legend.key = element_rect(fill = "white", color = NA),
      legend.title = element_blank(),
      legend.text = element_text(size = base_font_size),
      legend.key.width = grid::unit(0.95, "cm"),
      legend.spacing.x = grid::unit(0.18, "cm"),
      legend.margin = margin(4, 10, 4, 10),
      plot.margin = margin(8, 10, 8, 8)
    )
}

root <- project_root(script_dir())
reconciliation_root <- file.path(
  root,
  "hierarchical_reconciliation",
  "reconciliation_outputs"
)
dynamic_reconciliation_root <- file.path(
  root,
  "dynamic_weight_end_to_end_reconciliation",
  "frozen_dynamic_wls_outputs"
)
figure_dir <- file.path(root, "visualization", "reconciliation_plots", "plots")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

all_predictions <- lapply(target_order, function(target_name) {
  target_data <- read_target_predictions(
    reconciliation_root,
    dynamic_reconciliation_root,
    target_name
  )
  prediction_long(target_data)
})
plot_data <- do.call(rbind, all_predictions)

panel_plot <- make_panel_plot(plot_data)
output_png <- file.path(figure_dir, "lstm_kan_reconciliation_predictions_panel.png")
output_pdf <- file.path(figure_dir, "lstm_kan_reconciliation_predictions_panel.pdf")

ggsave(
  output_png,
  panel_plot,
  width = plot_width,
  height = plot_height,
  dpi = plot_dpi,
  bg = "white"
)
ggsave(
  output_pdf,
  panel_plot,
  width = plot_width,
  height = plot_height,
  device = grDevices::cairo_pdf,
  bg = "white"
)

message("Input directory: ", normalizePath(reconciliation_root, winslash = "/", mustWork = TRUE))
message("DynamicWLS directory: ", normalizePath(dynamic_reconciliation_root, winslash = "/", mustWork = TRUE))
message("Saved: ", normalizePath(output_png, winslash = "/", mustWork = TRUE))
message("Saved: ", normalizePath(output_pdf, winslash = "/", mustWork = TRUE))
