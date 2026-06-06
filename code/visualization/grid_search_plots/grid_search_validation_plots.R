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
# 本段控制图件字号、边框、颜色和导出尺寸；后续调图时优先改这里。
font_family <- "Arial"
base_font_size <- 15
panel_border_linewidth <- 1.00
strip_border_linewidth <- 0.75
line_width_observed <- 1.00
line_width_predicted <- 0.78
output_width <- 15.0
output_height <- 8.5
output_dpi <- 600

if (.Platform$OS.type == "windows") {
  grDevices::windowsFonts(Arial = grDevices::windowsFont("Arial"))
}

# 子图顺序与标题。只改变图中显示名，不改变读取文件时使用的目标列名。
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

# 模型颜色与全时段时间序列图保持一致，便于跨图比较。
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

# 验证集时间跨度较短，日期标签显示到月日，便于辨认局部波动。
date_breaks <- scales::breaks_pretty(n = 5)
date_labels <- scales::label_date("%Y-%m-%d")
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
    regexec("^(lstm_[a-z]+)_best_predictions_val\\.csv$", name)
  )[[1]]
  if (length(parts) == 0) {
    stop("Unexpected validation prediction filename: ", path)
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
    stop("Unknown model id in validation filename: ", data$model_id[1])
  }
  data$dataset_id <- "val"
  data
}

read_target_validation_predictions <- function(target_dir, target_name) {
  files <- list.files(
    target_dir,
    pattern = "^lstm_[a-z]+_best_predictions_val\\.csv$",
    full.names = TRUE
  )
  if (length(files) == 0) {
    stop("No validation prediction files found in: ", target_dir)
  }

  data <- do.call(rbind, lapply(files, read_prediction_file))
  data <- data[data$target == target_name, ]
  expected_models <- names(model_labels)
  missing_models <- setdiff(expected_models, unique(data$model_id))
  if (length(missing_models) > 0) {
    stop(
      "Missing validation prediction files for target ",
      target_name,
      ": ",
      paste(missing_models, collapse = ", ")
    )
  }

  data$model <- factor(data$model, levels = unname(model_labels))
  data[order(data$model, data$date), ]
}

observed_series <- function(predictions) {
  observed <- unique(predictions[, c("date", "target", "true_value")])
  names(observed)[names(observed) == "true_value"] <- "value"
  observed$series <- "Observed"
  observed
}

target_time_series <- function(target_name, predictions) {
  observed <- observed_series(predictions)
  predicted <- data.frame(
    date = predictions$date,
    target = predictions$target,
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

publication_theme <- function() {
  theme_bw(base_size = base_font_size, base_family = font_family) +
    theme(
      # 白色背景适合论文和汇报，避免导出后出现透明背景。
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      # 每个子图保留完整黑色外框，统一线宽便于和其他图件保持一致。
      panel.border = element_rect(fill = NA, color = "black", linewidth = panel_border_linewidth),
      # 验证集图保留浅灰横向网格，方便比较预测曲线和观测曲线的纵向偏差。
      panel.grid.major.y = element_line(color = "gray88", linewidth = 0.45),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.spacing = grid::unit(0.95, "lines"),
      # 灰色标题框用于明确绑定子图标题和对应面板。
      strip.background = element_rect(
        fill = "grey80",
        color = "black",
        linewidth = strip_border_linewidth
      ),
      strip.text = element_text(size = base_font_size, hjust = 0.5, face = "bold"),
      axis.title.y = element_text(size = base_font_size, margin = margin(r = 10)),
      axis.title.x = element_blank(),
      axis.text.x = element_text(
        size = base_font_size,
        angle = 45,
        hjust = 1,
        vjust = 1,
        color = "black"
      ),
      axis.text.y = element_text(size = base_font_size, color = "black"),
      axis.ticks = element_line(color = "black", linewidth = 0.35),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.justification = "center",
      legend.background = element_rect(fill = "white", color = "black", linewidth = 0.40),
      legend.box.background = element_blank(),
      legend.key = element_rect(fill = "white", color = NA),
      legend.title = element_blank(),
      legend.text = element_text(size = base_font_size),
      # 加宽图例线段和左右留白，避免较长模型名称挤出图例框。
      legend.key.width = grid::unit(1.55, "cm"),
      legend.margin = margin(5, 16, 5, 16),
      legend.box.margin = margin(4, 8, 4, 8),
      legend.spacing.x = grid::unit(0.42, "cm"),
      plot.margin = margin(8, 10, 8, 8)
    )
}

make_single_validation_plot <- function(plot_data, show_legend = FALSE) {
  ggplot(
    plot_data,
    aes(
      x = .data$date,
      y = .data$value,
      color = .data$series,
      linetype = .data$series,
      linewidth = .data$series,
      group = interaction(.data$target_panel, .data$series)
    )
  ) +
    geom_line(na.rm = TRUE) +
    facet_wrap(vars(.data$target_panel), ncol = 1, scales = "free_y") +
    scale_color_manual(values = series_colors, breaks = series_levels, drop = FALSE) +
    scale_linetype_manual(values = series_linetypes, breaks = series_levels, drop = FALSE) +
    scale_linewidth_manual(values = series_linewidths, breaks = series_levels, drop = FALSE) +
    scale_x_date(
      breaks = date_breaks,
      labels = date_labels,
      expand = expansion(mult = c(0.01, 0.02))
    ) +
    scale_y_continuous(
      breaks = value_breaks,
      labels = value_labels,
      expand = expansion(mult = c(0.08, 0.12))
    ) +
    labs(x = NULL, y = y_axis_label) +
    guides(
      color = guide_legend(
        nrow = 1,
        byrow = TRUE,
        override.aes = list(
          linetype = unname(series_linetypes[series_levels]),
          linewidth = unname(series_linewidths[series_levels])
        )
      ),
      linetype = "none",
      linewidth = "none"
    ) +
    publication_theme() +
    theme(
      # 手动拼图时，每个子图都保留横坐标刻度，便于单独读取验证集日期。
      legend.position = if (show_legend) "bottom" else "none"
    )
}

legend_plot <- function(plot_data) {
  make_single_validation_plot(plot_data, show_legend = TRUE)
}

extract_legend <- function(plot) {
  grob <- ggplotGrob(plot)
  idx <- which(vapply(grob$grobs, function(x) x$name, character(1)) == "guide-box")
  if (length(idx) == 0) {
    return(NULL)
  }
  grob$grobs[[idx[1]]]
}

make_validation_plots <- function(plot_data) {
  plots <- vector("list", length(target_order))
  names(plots) <- target_order
  for (target_name in target_order) {
    target_data <- plot_data[plot_data$target == target_name, ]
    plots[[target_name]] <- make_single_validation_plot(
      plot_data = target_data,
      show_legend = FALSE
    )
  }
  plots
}

draw_validation_panel <- function(plots, legend_grob = NULL) {
  grid::grid.newpage()
  layout <- grid::grid.layout(
    nrow = 3,
    ncol = 6,
    widths = grid::unit(rep(1, 6), "null"),
    heights = grid::unit.c(
      grid::unit(1, "null"),
      grid::unit(1, "null"),
      grid::unit(0.68, "in")
    )
  )
  grid::pushViewport(grid::viewport(layout = layout))

  # 上排三个子图平均占满整行；下排两个子图放在中间列位，避免左对齐造成视觉失衡。
  print(plots[["Green_Algae"]], vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 1:2))
  print(plots[["Cyanobacteria"]], vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 3:4))
  print(plots[["Diatoms"]], vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 5:6))
  print(plots[["Cryptophyta"]], vp = grid::viewport(layout.pos.row = 2, layout.pos.col = 2:3))
  print(plots[["Algae_Sum"]], vp = grid::viewport(layout.pos.row = 2, layout.pos.col = 4:5))

  # 图例放在整张图底部，横跨全部列，并预留更高区域防止文字超出图例框。
  if (!is.null(legend_grob)) {
    grid::pushViewport(grid::viewport(layout.pos.row = 3, layout.pos.col = 1:6))
    grid::grid.draw(legend_grob)
    grid::popViewport()
  }
  grid::popViewport()
}

save_validation_panel <- function(path, plots, legend_grob, width = output_width, height = output_height, dpi = output_dpi) {
  ext <- tolower(tools::file_ext(path))
  if (identical(ext, "png")) {
    grDevices::png(
      filename = path,
      width = width,
      height = height,
      units = "in",
      res = dpi,
      bg = "white",
      type = if (.Platform$OS.type == "windows") "windows" else "cairo"
    )
    on.exit(grDevices::dev.off(), add = TRUE)
    draw_validation_panel(plots, legend_grob)
    return(invisible(path))
  }
  if (identical(ext, "pdf")) {
    grDevices::cairo_pdf(filename = path, width = width, height = height, bg = "white")
    on.exit(grDevices::dev.off(), add = TRUE)
    draw_validation_panel(plots, legend_grob)
    return(invisible(path))
  }
  stop("Unsupported output extension: ", ext)
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
  predictions <- read_target_validation_predictions(
    file.path(prediction_root, target_name),
    target_name
  )
  target_time_series(target_name, predictions)
})
plot_data <- do.call(rbind, all_series)

plots <- make_validation_plots(plot_data)
legend_grob <- extract_legend(legend_plot(plot_data))
output_png <- file.path(figure_dir, "five_algae_validation_timeseries_panel.png")
output_pdf <- file.path(figure_dir, "five_algae_validation_timeseries_panel.pdf")

save_validation_panel(output_png, plots, legend_grob)
save_validation_panel(output_pdf, plots, legend_grob)

message("Input directory: ", normalizePath(prediction_root, winslash = "/", mustWork = TRUE))
message("Saved: ", normalizePath(output_png, winslash = "/", mustWork = TRUE))
message("Saved: ", normalizePath(output_pdf, winslash = "/", mustWork = TRUE))
