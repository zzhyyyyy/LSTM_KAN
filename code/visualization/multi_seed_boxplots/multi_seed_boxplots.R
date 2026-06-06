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

font_family <- "Arial"
if (.Platform$OS.type == "windows") {
  grDevices::windowsFonts(Arial = grDevices::windowsFont("Arial"))
}

# 箱线图使用测试集结果；如需切换数据集，可改为 "train" 或 "val"。
dataset_to_plot <- "test"

# 2 x 2 子图的指标顺序，保持和论文图中面板标号一致。
metrics_to_plot <- c("NSE", "KGE", "nMAE", "nRMSE")
metric_titles <- c(
  NSE = "NSE",
  KGE = "KGE",
  nMAE = "nMAE",
  nRMSE = "nRMSE"
)

# x 轴从左到右的预测目标顺序。
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

# 三种模型的显示顺序和论文友好的柔和配色。
model_order <- c("LSTM_FC", "LSTM_MLP", "LSTM_KAN")
model_labels <- c(
  LSTM_FC = "LSTM-FC",
  LSTM_MLP = "LSTM-MLP",
  LSTM_KAN = "LSTM-KAN"
)
# 箱体填充色：使用接近参考图的柔和配色。修改这些十六进制色值即可
# 同时改变箱线图填充色和图例色块。
model_fill_colors <- c(
  "LSTM-FC" = "#F5C9BC",
  "LSTM-MLP" = "#CEA1B5",
  "LSTM-KAN" = "#6F8CB8"
)

# 箱线边框和散点颜色：使用由填充色加深得到的非黑色描边，
# 避免原先黑色框线过重。
model_line_colors <- c(
  "LSTM-FC" = "#D08F82",
  "LSTM-MLP" = "#A8738C",
  "LSTM-KAN" = "#426C9B"
)

# 主要绘图样式参数：集中放在这里，后续微调图形细节时不需要到
# ggplot 图层里逐项查找。
box_width <- 0.68              # 每个箱体的宽度。
dodge_width <- 0.7            # 同一预测目标内，不同模型之间的横向间距。
whisker_cap_width <- 0.28      # 上下极限横线的宽度。
boxplot_coef <- 1.5            # 箱线图须线范围；超过 1.5 倍 IQR 的点会作为异常值显示。
box_linewidth <- 0.50          # 箱体边框、须线和中位数线的线宽。
box_alpha <- 1.00              # 箱体填充透明度。
outlier_point_size <- 2.00     # 异常值散点大小；只影响箱线图识别出的异常值。
outlier_point_alpha <- 1.00    # 异常值散点透明度。
panel_border_linewidth <- 1.00 # 每个子图黑色外边框线宽。
strip_fill <- "grey80"         # 子图标题灰色背景。

# y 轴刻度：仅对指定面板手动设置刻度，其余面板仍按数据分布自动生成。
# 当前 2 x 2 面板顺序为：
# (a) NSE、(b) KGE、(c) nMAE、(d) nRMSE。
# 由于 ggplot2 的 free_y 分面刻度函数只接收当前面板的 y 轴范围，
# 这里通过当前数据范围识别 (a) 与 (d)，从而保持其他面板设置不变。
value_breaks <- function(limits) {
  finite_limits <- limits[is.finite(limits)]
  if (length(finite_limits) == 0) {
    return(numeric(0))
  }

  y_range <- range(finite_limits, na.rm = TRUE)

  # (d) nRMSE 面板：数据上界超过 1，按 0.3、0.6、0.9、1.2 显示。
  if (y_range[2] > 1.0) {
    return(c(0.3, 0.6, 0.9, 1.2))
  }

  # (a) NSE 面板：当前范围集中在 0.4、0.6、0.8 附近，手动固定三档。
  if (y_range[1] > 0.25 && y_range[2] < 0.84) {
    return(c(0.4, 0.6, 0.8))
  }

  scales::breaks_pretty(n = 4)(limits)
}

# y 轴刻度格式：保留真实尺度，只优化显示文本，不缩放或删除数据。
format_axis_number <- function(x) {
  output <- rep("NA", length(x))
  valid <- !is.na(x) & is.finite(x)
  output[valid] <- scales::number(
    x[valid],
    accuracy = 0.01,
    trim = TRUE
  )
  output
}

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
    "base_model",
    "single_output_model",
    "multi_seed_outputs",
    "metrics",
    "best_params_multi_seed_metrics.csv"
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

prepare_plot_data <- function(metrics_path) {
  data <- read.csv(metrics_path, stringsAsFactors = FALSE, check.names = FALSE)
  data <- data[data$dataset == dataset_to_plot, ]
  if (nrow(data) == 0) {
    stop("No rows found for dataset: ", dataset_to_plot)
  }

  data <- data[data$model_name %in% model_order, ]
  data$model <- factor(
    model_labels[data$model_name],
    levels = unname(model_labels)
  )

  available_targets <- target_order[target_order %in% data$target]
  extra_targets <- setdiff(unique(data$target), target_order)
  target_levels <- c(available_targets, sort(extra_targets))
  display_labels <- target_labels[target_levels]
  display_labels[is.na(display_labels)] <- target_levels[is.na(display_labels)]

  data$target <- factor(data$target, levels = target_levels)
  data$target_label <- factor(
    display_labels[as.character(data$target)],
    levels = display_labels
  )
  data
}

add_normalized_metrics <- function(data, root) {
  data_path <- file.path(
    root,
    "base_model",
    "data",
    paste0(dataset_to_plot, "_model_input.csv")
  )
  observed <- read.csv(data_path, stringsAsFactors = FALSE, check.names = FALSE)

  target_names <- as.character(unique(data$target))
  target_means <- setNames(rep(NA_real_, length(target_names)), target_names)
  for (target in names(target_means)) {
    if (target %in% names(observed)) {
      values <- observed[[target]]
    } else {
      target_col <- unique(data$target_col[data$target == target])[1]
      if (!target_col %in% names(observed)) {
        stop("Cannot find observed values for target: ", target)
      }
      values <- expm1(observed[[target_col]])
    }
    target_means[[target]] <- mean(values, na.rm = TRUE)
  }

  data$target_mean <- target_means[as.character(data$target)]
  data$nRMSE <- data$RMSE / data$target_mean
  data$nMAE <- data$MAE / data$target_mean
  data
}

make_long_data <- function(data, metrics) {
  rows <- lapply(metrics, function(metric) {
    metric_data <- data[is.finite(data[[metric]]), ]
    data.frame(
      target = metric_data$target_label,
      model = metric_data$model,
      metric = metric,
      value = metric_data[[metric]],
      seed = metric_data$seed,
      stringsAsFactors = FALSE
    )
  })
  long_data <- do.call(rbind, rows)
  if (nrow(long_data) == 0) {
    stop("No finite values found for metrics: ", paste(metrics, collapse = ", "))
  }

  panel_labels <- setNames(
    paste0("(", letters[seq_along(metrics)], ") ", metric_titles[metrics]),
    metrics
  )
  long_data$target <- factor(long_data$target, levels = target_labels[target_order])
  long_data$model <- factor(long_data$model, levels = unname(model_labels))
  long_data$metric <- factor(
    panel_labels[long_data$metric],
    levels = unname(panel_labels)
  )
  long_data$seed <- as.factor(long_data$seed)
  long_data
}

# 论文/汇报图主题：
# - 图例放在整图底部；
# - 子图标题位于子图上方并居中，标题背景为灰色并添加黑色外边框；
# - 保留原来的横向浅灰内部网格线，便于读取纵坐标；
# - 每个子图保留 1.00 pt 黑色实线外边框。
plot_theme <- theme_bw(base_size = 15, base_family = font_family) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    panel.border = element_rect(
      fill = NA,
      color = "black",
      linewidth = panel_border_linewidth
    ),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_line(color = "gray88", linewidth = 0.45),
    panel.grid.minor.y = element_blank(),
    strip.background = element_rect(
      fill = strip_fill,
      color = "black",
      linewidth = panel_border_linewidth
    ),
    strip.text = element_text(
      size = 15,
      hjust = 0.5,
      margin = margin(t = 6, b = 6)
    ),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_text(size = 15, color = "black", angle = 25, hjust = 1),
    axis.text.y = element_text(size = 15, color = "black"),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.justification = "center",
    legend.background = element_blank(),
    legend.box.background = element_blank(),
    legend.key = element_rect(fill = "white", color = NA),
    legend.title = element_blank(),
    legend.text = element_text(size = 15),
    legend.key.size = grid::unit(0.42, "cm"),
    panel.spacing = grid::unit(1.0, "lines"),
    plot.margin = margin(8, 10, 8, 10)
  )

make_optimized_plot <- function(plot_data) {
  # position_dodge 用于拉开同一藻类目标下不同模型的箱体位置。
  dodge <- position_dodge(width = dodge_width)

  # 图层顺序：
  # 1. 先绘制上下极限横线，让箱线图的须线端点更清晰；
  # 2. 再绘制箱线图主体，并只显示箱线图判定出的异常值散点；
  # 3. 最后通过 facet_wrap 生成四个指标子图。
  ggplot(
    plot_data,
    aes(x = .data$target, y = .data$value, fill = .data$model)
  ) +
    stat_boxplot(
      aes(
        color = .data$model,
        group = interaction(.data$target, .data$model)
      ),
      geom = "errorbar",
      width = whisker_cap_width,
      position = dodge,
      coef = boxplot_coef,
      linewidth = box_linewidth
    ) +
    geom_boxplot(
      aes(
        color = .data$model,
        group = interaction(.data$target, .data$model)
      ),
      width = box_width,
      position = dodge,
      coef = boxplot_coef,
      outlier.shape = 16,
      outlier.size = outlier_point_size,
      outlier.alpha = outlier_point_alpha,
      linewidth = box_linewidth,
      alpha = box_alpha
    ) +
    scale_fill_manual(values = model_fill_colors, drop = FALSE) +
    scale_color_manual(values = model_line_colors, drop = FALSE, guide = "none") +
    scale_y_continuous(
      breaks = value_breaks,
      labels = format_axis_number,
      expand = expansion(mult = c(0.08, 0.12))
    ) +
    facet_wrap(
      ~ metric,
      ncol = 2,
      scales = "free_y"
    ) +
    labs(x = NULL, y = NULL) +
    plot_theme +
    guides(
      fill = guide_legend(
        nrow = 1,
        byrow = TRUE,
        override.aes = list(alpha = box_alpha, color = NA)
      )
    )
}

root <- project_root(script_dir())
metrics_path <- file.path(
  root,
  "base_model",
  "single_output_model",
  "multi_seed_outputs",
  "metrics",
  "best_params_multi_seed_metrics.csv"
)

# 输出目录沿用 multi_seed_boxplots/plots，不覆盖任何原始数值文件。
figure_dir <- file.path(root, "visualization", "multi_seed_boxplots", "plots")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

plot_data <- prepare_plot_data(metrics_path)
plot_data <- add_normalized_metrics(plot_data, root)
plot_long <- make_long_data(plot_data, metrics_to_plot)
optimized_plot <- make_optimized_plot(plot_long)

pdf_path <- file.path(figure_dir, "model_performance_boxplot_optimized.pdf")
png_path <- file.path(figure_dir, "model_performance_boxplot_optimized.png")

ggsave(
  pdf_path,
  optimized_plot,
  width = 11,
  height = 8.5,
  device = grDevices::cairo_pdf,
  bg = "white"
)
ggsave(
  png_path,
  optimized_plot,
  width = 11,
  height = 8.5,
  dpi = 600,
  bg = "white"
)

message("Saved: ", normalizePath(pdf_path, winslash = "/", mustWork = TRUE))
message("Saved: ", normalizePath(png_path, winslash = "/", mustWork = TRUE))
