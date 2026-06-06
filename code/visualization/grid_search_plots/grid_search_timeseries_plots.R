if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("Package 'ggplot2' is required. Install it with install.packages('ggplot2').")
}
if (!requireNamespace("scales", quietly = TRUE)) {
  stop("Package 'scales' is required. Install it with install.packages('scales').")
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

invisible(Sys.setlocale("LC_TIME", "C"))

# =========================
# 基础显示设置
# =========================
# 这里集中控制全图字体、线宽、颜色和导出尺寸，后续美化时优先修改这一段。
font_family <- "Arial"
base_font_size <- 15
panel_border_linewidth <- 1.00
strip_border_linewidth <- 0.75
line_width_observed <- 0.95
line_width_predicted <- 0.75
output_width <- 11.6
output_height <- 13.4
output_dpi <- 600

if (.Platform$OS.type == "windows") {
  grDevices::windowsFonts(Arial = grDevices::windowsFont("Arial"))
}

# 子图顺序与标题。图中只显示规范英文名，不改变原始数据列名。
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

# 模型颜色在项目图件中保持一致；Observed 用深灰，三个模型用高对比柔和色。
model_labels <- c(
  lstm_fc = "LSTM-FC",
  lstm_mlp = "LSTM-MLP",
  lstm_kan = "LSTM-KAN"
)
series_levels <- c("Observed", unname(model_labels))
series_colors <- c(
  "Observed" = "#222222",
  "LSTM-FC" = "#D08F82",
  "LSTM-MLP" = "#426C9B",
  "LSTM-KAN" = "#A8738C"
)
series_linetypes <- c(
  "Observed" = "solid",
  "LSTM-FC" = "22",
  "LSTM-MLP" = "22",
  "LSTM-KAN" = "22"
)
series_linewidths <- c(
  "Observed" = line_width_observed,
  "LSTM-FC" = line_width_predicted,
  "LSTM-MLP" = line_width_predicted,
  "LSTM-KAN" = line_width_predicted
)
series_alpha <- c(
  "Observed" = 1.00,
  "LSTM-FC" = 0.92,
  "LSTM-MLP" = 0.92,
  "LSTM-KAN" = 0.92
)

# 坐标轴刻度设置。x 轴按年份显示，y 轴根据各子图数据自动给出 4 个左右刻度。
date_breaks <- scales::breaks_width("1 year")
date_labels <- scales::label_date("%Y")
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
    "base_model",
    "single_output_model",
    "grid_search_outputs",
    "predictions"
  )
  while (!dir.exists(file.path(current, marker))) {
    parent <- dirname(current)
    if (identical(parent, current)) {
      stop("Could not find project root containing: ", marker)
    }
    current <- parent
  }
  current
}

read_prediction_file <- function(path) {
  name <- basename(path)
  parts <- regmatches(
    name,
    regexec("^(lstm_[a-z]+)_best_predictions_(train|val|test)\\.csv$", name)
  )[[1]]
  if (length(parts) == 0) {
    stop("Unexpected prediction filename: ", path)
  }

  data <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  required_cols <- c("date", "target", "true_value", "pred_value")
  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0) {
    stop("Missing columns in ", path, ": ", paste(missing_cols, collapse = ", "))
  }

  data$date <- as.Date(data$date)
  data$true_value <- as.numeric(data$true_value)
  data$pred_value <- as.numeric(data$pred_value)
  data$model_id <- parts[2]
  data$model <- unname(model_labels[data$model_id])
  if (is.na(data$model[1])) {
    stop("Unknown model id in prediction filename: ", data$model_id[1])
  }
  data$dataset_id <- parts[3]
  data
}

read_target_predictions <- function(target_dir, target_name) {
  files <- list.files(
    target_dir,
    pattern = "^lstm_[a-z]+_best_predictions_(train|val|test)\\.csv$",
    full.names = TRUE
  )
  if (length(files) == 0) {
    stop("No prediction files found in: ", target_dir)
  }

  data <- do.call(rbind, lapply(files, read_prediction_file))
  data <- data[data$target == target_name, ]
  expected_models <- names(model_labels)
  expected_datasets <- c("train", "val", "test")
  missing_files <- expand.grid(
    model_id = expected_models,
    dataset_id = expected_datasets,
    stringsAsFactors = FALSE
  )
  missing_files$key <- paste(missing_files$model_id, missing_files$dataset_id)
  data$key <- paste(data$model_id, data$dataset_id)
  missing_keys <- setdiff(missing_files$key, unique(data$key))
  if (length(missing_keys) > 0) {
    stop("Missing prediction files for ", target_name, ": ", paste(missing_keys, collapse = ", "))
  }

  data$model <- factor(data$model, levels = unname(model_labels))
  data[order(data$model, data$dataset_id, data$date), ]
}

observed_series <- function(predictions) {
  observed <- unique(
    predictions[, c("date", "target", "dataset_id", "true_value")]
  )
  names(observed)[names(observed) == "true_value"] <- "value"
  observed$series <- "Observed"
  observed
}

target_time_series <- function(target_name, predictions) {
  observed <- observed_series(predictions)
  predicted <- data.frame(
    date = predictions$date,
    target = predictions$target,
    dataset_id = predictions$dataset_id,
    value = predictions$pred_value,
    series = as.character(predictions$model),
    stringsAsFactors = FALSE
  )

  data <- rbind(observed[, names(predicted)], predicted)
  data$target_panel <- factor(
    target_labels[data$target],
    levels = target_labels[target_order]
  )
  data$series <- factor(data$series, levels = series_levels)
  data
}

dataset_ranges <- function(plot_data) {
  ranges <- lapply(c("train", "val", "test"), function(dataset_name) {
    dates <- plot_data$date[plot_data$dataset_id == dataset_name]
    if (length(dates) == 0 || all(is.na(dates))) {
      stop("No dates found for dataset: ", dataset_name)
    }
    data.frame(
      dataset_id = dataset_name,
      xmin = min(dates, na.rm = TRUE),
      xmax = max(dates, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  ranges <- do.call(rbind, ranges)
  ranges$label <- c("Training set", "Validation set", "Testing set")
  ranges
}

publication_theme <- function() {
  theme_bw(base_size = base_font_size, base_family = font_family) +
    theme(
      # 背景保持白色，便于论文、PPT 和 PDF 排版。
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      # 保留完整子图外框，线宽统一为 1.00，增强多面板结构感。
      panel.border = element_rect(fill = NA, color = "black", linewidth = panel_border_linewidth),
      # 时间序列只保留浅灰色横向主网格，辅助读取纵坐标，不增加纵向干扰线。
      panel.grid.major.y = element_line(color = "gray88", linewidth = 0.45),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.spacing = grid::unit(0.72, "lines"),
      # 灰色标题框对应每个藻类子图，标题居中且字体统一。
      strip.background = element_rect(
        fill = "grey80",
        color = "black",
        linewidth = strip_border_linewidth
      ),
      strip.text = element_text(size = base_font_size, hjust = 0.5, face = "bold"),
      axis.title.y = element_text(size = base_font_size, margin = margin(r = 10)),
      axis.title.x = element_blank(),
      axis.text = element_text(size = base_font_size, color = "black"),
      axis.ticks = element_line(color = "black", linewidth = 0.35),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.justification = "center",
      legend.background = element_rect(fill = "white", color = "black", linewidth = 0.40),
      legend.box.background = element_blank(),
      legend.key = element_rect(fill = "white", color = NA),
      legend.title = element_blank(),
      legend.text = element_text(size = base_font_size),
      legend.key.width = grid::unit(0.90, "cm"),
      legend.spacing.x = grid::unit(0.20, "cm"),
      plot.margin = margin(10, 12, 10, 10)
    )
}

make_timeseries_panel <- function(plot_data) {
  x_limits <- range(plot_data$date, finite = TRUE)
  ranges <- dataset_ranges(plot_data)
  split_dates <- as.Date(c(
    ranges$xmax[ranges$dataset_id == "train"],
    ranges$xmax[ranges$dataset_id == "val"]
  ))

  top_panel <- factor(
    target_labels[target_order[1]],
    levels = target_labels[target_order]
  )
  label_data <- ranges
  label_data$x <- as.Date(
    rowMeans(cbind(as.numeric(label_data$xmin), as.numeric(label_data$xmax))),
    origin = "1970-01-01"
  )
  label_data$target_panel <- top_panel
  label_data$y <- Inf

  # 仅轻微高亮测试集区域，帮助区分最终评估阶段，同时避免遮挡曲线。
  test_region <- ranges[ranges$dataset_id == "test", c("xmin", "xmax")]

  ggplot(
    plot_data,
    aes(
      x = .data$date,
      y = .data$value,
      color = .data$series,
      linetype = .data$series,
      linewidth = .data$series,
      alpha = .data$series,
      group = interaction(.data$target_panel, .data$series)
    )
  ) +
    geom_rect(
      data = test_region,
      aes(xmin = .data$xmin, xmax = .data$xmax, ymin = -Inf, ymax = Inf),
      inherit.aes = FALSE,
      fill = "#E7F1EF",
      alpha = 0.16,
      color = NA
    ) +
    # 虚线标注训练/验证/测试分界，线宽较细以免喧宾夺主。
    geom_vline(
      xintercept = as.numeric(split_dates),
      color = "gray55",
      linetype = "22",
      linewidth = 0.45
    ) +
    geom_line(na.rm = TRUE) +
    # 数据集阶段标签只放在最上方子图，避免每个面板重复标注。
    geom_text(
      data = label_data,
      aes(x = .data$x, y = .data$y, label = .data$label),
      inherit.aes = FALSE,
      family = font_family,
      size = base_font_size / 2.845276,
      vjust = 1.35,
      color = "gray25"
    ) +
    facet_wrap(vars(.data$target_panel), ncol = 1, scales = "free_y") +
    scale_color_manual(values = series_colors, breaks = series_levels, drop = FALSE) +
    scale_linetype_manual(values = series_linetypes, breaks = series_levels, drop = FALSE) +
    scale_linewidth_manual(values = series_linewidths, breaks = series_levels, drop = FALSE) +
    scale_alpha_manual(values = series_alpha, breaks = series_levels, guide = "none") +
    scale_x_date(
      limits = x_limits,
      breaks = date_breaks,
      labels = date_labels,
      expand = expansion(mult = c(0.004, 0.01))
    ) +
    scale_y_continuous(
      breaks = value_breaks,
      labels = value_labels,
      expand = expansion(mult = c(0.08, 0.18))
    ) +
    labs(x = NULL, y = y_axis_label) +
    guides(
      color = guide_legend(
        nrow = 1,
        byrow = TRUE,
        override.aes = list(
          linetype = unname(series_linetypes[series_levels]),
          linewidth = unname(series_linewidths[series_levels]),
          alpha = unname(series_alpha[series_levels])
        )
      ),
      linetype = "none",
      linewidth = "none"
    ) +
    publication_theme()
}

root <- project_root(script_dir())
prediction_root <- file.path(
  root,
  "base_model",
  "single_output_model",
  "grid_search_outputs",
  "predictions"
)
figure_dir <- file.path(root, "visualization", "grid_search_plots", "plots")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

all_series <- lapply(target_order, function(target_name) {
  predictions <- read_target_predictions(
    file.path(prediction_root, target_name),
    target_name
  )
  target_time_series(target_name, predictions)
})
plot_data <- do.call(rbind, all_series)

timeseries_panel <- make_timeseries_panel(plot_data)
output_png <- file.path(figure_dir, "five_algae_timeseries_refined.png")
output_pdf <- file.path(figure_dir, "five_algae_timeseries_refined.pdf")

ggsave(
  output_png,
  timeseries_panel,
  width = output_width,
  height = output_height,
  dpi = output_dpi,
  bg = "white"
)
ggsave(
  output_pdf,
  timeseries_panel,
  width = output_width,
  height = output_height,
  device = grDevices::cairo_pdf,
  bg = "white"
)

message("Input directory: ", normalizePath(prediction_root, winslash = "/", mustWork = TRUE))
message("Saved: ", normalizePath(output_png, winslash = "/", mustWork = TRUE))
message("Saved: ", normalizePath(output_pdf, winslash = "/", mustWork = TRUE))
