import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf

# =========================
# 1. 路径与基础设置
# =========================
input_path = "code/data_pre/raw_data.csv"
output_dir = "code/data_pre/raw_data_EDA_outputs"
os.makedirs(output_dir, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 400
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 18
sns.set_style("whitegrid")

# =========================
# 2. 读取数据
# =========================
df = pd.read_csv(input_path)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 如果原始数据中有Bluegreen_Algae，则重命名为Cyanobacteria
if "Bluegreen_Algae" in df.columns and "Cyanobacteria" not in df.columns:
    df.rename(columns={"Bluegreen_Algae": "Cyanobacteria"}, inplace=True)

# 重命名温度列
rename_cols = {}
if "WTemperature" in df.columns and "WT" not in df.columns:
    rename_cols["WTemperature"] = "WT"
if "ATemperature" in df.columns and "AT" not in df.columns:
    rename_cols["ATemperature"] = "AT"
if rename_cols:
    df.rename(columns=rename_cols, inplace=True)

# =========================
# 3. 构造新的总藻量
# =========================
algae_cols = ["Green_Algae", "Cyanobacteria", "Diatoms", "Cryptophyta"]
df["Algae_Sum"] = df[algae_cols].sum(axis=1)
df["Algae_diff"] = df["Algae_Total"] - df["Algae_Sum"]

# =========================
# 4. 数值变量列表
# =========================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# =========================
# 5. 基础统计表
#    缺失值、均值、众数、分位数等
# =========================
stats_df = pd.DataFrame(index=numeric_cols)

stats_df["missing_count"] = df[numeric_cols].isnull().sum()
stats_df["missing_ratio_%"] = (df[numeric_cols].isnull().sum() / len(df) * 100).round(2)
stats_df["mean"] = df[numeric_cols].mean()
stats_df["std"] = df[numeric_cols].std()
stats_df["min"] = df[numeric_cols].min()
stats_df["25%"] = df[numeric_cols].quantile(0.25)
stats_df["50%_median"] = df[numeric_cols].quantile(0.5)
stats_df["75%"] = df[numeric_cols].quantile(0.75)
stats_df["max"] = df[numeric_cols].max()
stats_df["skewness"] = df[numeric_cols].skew().round(4)

# 众数：若有多个众数，只取第一个
modes = []
for col in numeric_cols:
    mode_series = df[col].mode(dropna=True)
    modes.append(mode_series.iloc[0] if not mode_series.empty else np.nan)
stats_df["mode"] = modes

# =========================
# 6. 离群值统计（IQR）
# =========================
outlier_count_list = []
outlier_ratio_list = []
lower_bound_list = []
upper_bound_list = []

for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_ratio = round(outlier_count / len(df) * 100, 2)

    lower_bound_list.append(lower)
    upper_bound_list.append(upper)
    outlier_count_list.append(outlier_count)
    outlier_ratio_list.append(outlier_ratio)

stats_df["outlier_count"] = outlier_count_list
stats_df["outlier_ratio_%"] = outlier_ratio_list
stats_df["iqr_lower"] = lower_bound_list
stats_df["iqr_upper"] = upper_bound_list

# 保存统计表
stats_df.to_csv(os.path.join(output_dir, "EDA_basic_statistics.csv"), encoding="utf-8-sig")

# =========================
# 7. 输出Algae_Total与Algae_Sum差异统计
# =========================
algae_compare = df[["date", "Algae_Total", "Algae_Sum", "Algae_diff"]]
algae_compare.to_csv(os.path.join(output_dir, "Algae_Total_vs_Algae_Sum.csv"), index=False, encoding="utf-8-sig")

algae_diff_summary = df["Algae_diff"].describe()
algae_diff_summary.to_csv(os.path.join(output_dir, "Algae_diff_summary.csv"), encoding="utf-8-sig")

# =========================
# 8. 藻类时间序列图
#    4种藻类放在一张图（不包括Algae_Total、Algae_Sum）
# =========================
plot_algae_cols = ["Green_Algae", "Cyanobacteria", "Diatoms", "Cryptophyta"]

fig, axes = plt.subplots(2, 2, figsize=(24, 14), sharex=True)
axes = axes.flatten()

for ax, col in zip(axes, plot_algae_cols):
    ax.plot(df["date"], df[col], linewidth=1.1, color="#4C72B0")
    ax.set_title(f"{col} time series", pad=12, fontweight="bold")
    ax.set_xlabel("Datetime")
    ax.set_ylabel(col)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.35)

fig.suptitle("Algae Time Series", fontsize=24, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(output_dir, "algae_time_series_combined.png"), bbox_inches="tight")
plt.close()

summary_algae_cols = ["Algae_Total", "Algae_Sum"]
summary_algae_cols = [col for col in summary_algae_cols if col in df.columns]

fig, axes = plt.subplots(2, 1, figsize=(24, 12), sharex=True)
axes = np.array(axes).flatten()

for ax, col in zip(axes, summary_algae_cols):
    ax.plot(df["date"], df[col], linewidth=1.2, color="#C44E52")
    ax.set_title(f"{col} time series", pad=12, fontweight="bold")
    ax.set_xlabel("Datetime")
    ax.set_ylabel(col)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.35)

fig.suptitle("Algae_Total and Algae_Sum Time Series", fontsize=24, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(output_dir, "algae_total_sum_time_series_combined.png"), bbox_inches="tight")
plt.close()

# =========================
# 9. 环境因子时序图
# =========================
env_candidate_cols = [
    "DO",
    "pH",
    "Turbidity",
    "Conductivity",
    "WT",
    "windspeed",
    "AT",
    "max_TEM",
    "min_TEM",
    "humidity",
    "precipitation",
    "NOX",
    "NH4",
    "DIN",
    "DIP",
    "TN",
    "TP",
    "NPR",
]
env_cols = [col for col in env_candidate_cols if col in df.columns]

env_output_dir = os.path.join(output_dir, "environment_time_series")
os.makedirs(env_output_dir, exist_ok=True)

env_group_1 = [
    "WT", "windspeed", "AT", "max_TEM",
    "min_TEM", "humidity", "precipitation", "DO"
]
env_group_2 = [col for col in env_cols if col not in env_group_1]
env_group_3 = [
    "WT", "windspeed", "AT", "pH",
    "TN", "TP", "Conductivity", "DO"
]

env_groups = [
    ("environment_time_series_group_1.png", "Environmental Factors Time Series - Group 1", env_group_1, 4, 2, (24, 18)),
    ("environment_time_series_group_2.png", "Environmental Factors Time Series - Group 2", env_group_2, 5, 2, (24, 22)),
    ("environment_time_series_main_variables.png", "Main Environmental Variables Time Series", env_group_3, 4, 2, (24, 18)),
]


env_summary_rows = []
for fig_name, fig_title, group_cols, n_rows, n_cols, fig_size in env_groups:
    group_cols = [col for col in group_cols if col in df.columns]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, sharex=True)
    axes = np.array(axes).flatten()

    for ax, col in zip(axes, group_cols):
        ax.plot(df["date"], df[col], linewidth=1.1, color="#4C72B0")
        ax.set_title(f"{col} time series", pad=10, fontweight="bold")
        ax.set_xlabel("Datetime")
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.35)
        env_summary_rows.append({
            "figure": fig_name,
            "variable": col,
        })

    for ax in axes[len(group_cols):]:
        fig.delaxes(ax)

    fig.suptitle(fig_title, fontsize=24, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(os.path.join(env_output_dir, fig_name), bbox_inches="tight")
    plt.close()

pd.DataFrame(env_summary_rows).to_csv(
    os.path.join(env_output_dir, "environment_time_series_figure_index.csv"),
    index=False,
    encoding="utf-8-sig",
)

# =========================
# 10. 所有藻类ACF图
# =========================
acf_cols = ["Green_Algae", "Cyanobacteria", "Diatoms", "Cryptophyta", "Algae_Total", "Algae_Sum"]
acf_cols = [col for col in acf_cols if col in df.columns]
max_lag = min(200, len(df) - 1)

fig, axes = plt.subplots(3, 2, figsize=(24, 24))
axes = axes.flatten()

for ax, col in zip(axes, acf_cols):
    series = df[col].dropna().to_numpy()
    col_lag = min(max_lag, len(series) - 1)
    acf_values = acf(series, nlags=col_lag, fft=True)
    lags = np.arange(len(acf_values))

    ax.axhline(0, color="0.2", linewidth=1.0)
    ax.vlines(lags, 0, acf_values, color="#005BBB", linewidth=1.6)
    ax.scatter(lags, acf_values, color="#E66100", s=28, zorder=3)
    ax.set_title(f"Autocorrelation Function (ACF) of {col}", pad=12, fontweight="bold")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_xlim(0, col_lag)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.8)

for ax in axes[len(acf_cols):]:
    fig.delaxes(ax)

fig.suptitle("ACF of Algae Variables", fontsize=26, fontweight="bold", y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(os.path.join(output_dir, "algae_acf_combined.png"), bbox_inches="tight")
plt.close()

# =========================
# 11. 相关性热图
#    简洁版：藻类 + 核心水环境变量
# =========================
corr_cols = [
    "Green_Algae", "Cyanobacteria", "Diatoms", "Cryptophyta",
    "Algae_Sum",
    "WT", "DO", "pH", "TN", "TP"
]

corr_df = df[corr_cols].corr(method="spearman")

plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", center=0, fmt=".2f", square=True)
plt.title("Spearman correlation heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300)
plt.close()

# 同时保存相关矩阵
corr_df.to_csv(os.path.join(output_dir, "correlation_matrix.csv"), encoding="utf-8-sig")

# All environmental factors correlation heatmap
env_corr_df = df[env_cols].corr(method="spearman")

plt.figure(figsize=(14, 12))
sns.heatmap(
    env_corr_df,
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f",
    square=True,
    annot_kws={"size": 7},
)
plt.title("Spearman correlation heatmap - all environmental factors")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "environment_correlation_heatmap.png"), dpi=300)
plt.close()

env_corr_df.to_csv(
    os.path.join(output_dir, "environment_correlation_matrix.csv"),
    encoding="utf-8-sig",
)

# =========================
# 12. 偏态分析可视化
# =========================
skewness = df[numeric_cols].skew().round(4)

plt.figure(figsize=(12, 6))
sns.barplot(x=skewness.index, y=skewness.values, hue=skewness.index, palette="Spectral", dodge=False, legend=False)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Skewness")
plt.title("Numeric variable skewness")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "skewness_barplot.png"), dpi=300)
plt.close()

hist_cols = [col for col in numeric_cols if col != "Algae_diff"]
hist_groups = np.array_split(hist_cols, 2)

for idx, group_cols in enumerate(hist_groups, start=1):
    group_cols = list(group_cols)
    n_cols = 3
    n_rows = int(np.ceil(len(group_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5.8 * n_rows))
    axes = np.array(axes).flatten()

    for ax, col in zip(axes, group_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, bins=30, color="steelblue")
        ax.set_title(f"{col}\nSkew={skewness[col]:.4f}", fontweight="bold", pad=10)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.tick_params(axis="x", rotation=25)

    for ax in axes[len(group_cols):]:
        fig.delaxes(ax)

    fig.suptitle(f"Numeric Skewness Histograms - Part {idx}", fontsize=24, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(os.path.join(output_dir, f"numeric_skewness_histograms_part_{idx}.png"), bbox_inches="tight")
    plt.close()

print("EDA completed. Results saved to:", output_dir)
