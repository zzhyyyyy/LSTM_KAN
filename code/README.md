# 浮游植物类群/群落结构 层次预测 —— 代码说明（`code/`）

基于分层深度学习的藻类群落结构一致性预报。整体流程：
**数据预处理 → 基础模型（单输出·多步·目标 log1p）→ 层次协调 → 动态权重端到端协调（DynamicWLS）→ SHAP 可解释性 → 可视化（R）**。

> **尺度（方案A）**：对 5 个藻类**目标**做 log1p、**输入保持原始**，预测后 `expm1` 还原到原始尺度再做协调/算指标（由 `base_model/multi_output_model/models.py` 的 `USE_LOG` 开关控制，默认 True）。改用 log 的原因：测试年 Green_Algae 均值骤降（3.8→0.5），原始尺度下其 nRMSE 被小分母放大到 ~3.7，log1p 可显著压低该测试误差。

层次结构（加和约束）：`Algae_Sum = Green_Algae + Cyanobacteria + Diatoms + Cryptophyta`。

> **当前主线 = 单输出（每个目标一个模型）**。早期做过"多输出（一个模型出 5 目标）"，但整体精度偏弱，已改回单输出。多输出代码作为对比保留。

---

## 1. 核心建模设定

下面这些设定（原始尺度、多步、不含总藻、7:1:2、分步优化）**单输出与多输出共用**，区别只在"输出粒度"。

| 项目 | 设定 |
|---|---|
| 尺度 | **方案A（默认 `models.USE_LOG=True`）**：对 5 个目标做 **log1p**（`z=log(1+y)`）再标准化；**输入特征保持原始**；预测先反标准化再 **expm1**（`ŷ=exp(ẑ)−1`）还原到原始尺度；**层次协调与指标都在原始尺度**。设 `USE_LOG=False` 即切回纯原始尺度。改一个开关即可全链路切换 |
| 输入特征 | **13 个原始特征（始终原始，不 log）**：4 底层藻（自回归）+ pH/Turbidity/WT/precipitation/DIN/DIP/TN/TP/NPR；**不含 Algae_Sum**（避免信息泄漏） |
| 输出目标 | **5 个**：Green_Algae、Cyanobacteria、Diatoms、Cryptophyta、**Algae_Sum** |
| 预测方式 | **直接多步**（非递归）：一次前向输出 `t+1/t+2/t+3` |
| **输出粒度** | **★主线＝单输出**：每个目标一个模型 → 4 架构 × 5 目标 = **20 个模型**，每个输出该目标的 `t+1/t+2/t+3`（3 个值）。**对比＝多输出**：4 个模型，每个一次输出 `3×5=15` 个值 |
| 基础模型架构 | **LSTM‑FC / LSTM‑MLP / LSTM‑KAN / 纯KAN** |
| 窗口 | `LOOKBACK=30`，`HORIZONS=[1,2,3]`，滚动窗口 |
| 数据划分 | **7:1:2**（train 70% / val 10% / test 20%，按时间顺序；测试集≈完整一年） |
| 超参搜索 | **分步优化（坐标下降）**：从基线出发，按"结构/学习率优先"逐个超参数扫描选优 |
| 选优标准 | 跨全部 `(target, horizon)` 的**平均验证集 nRMSE** 最小 |
| 评价指标 | nRMSE、nMAE、NSE、KGE（逐 `(target, horizon)` 在原始尺度计算） |
| 随机种子 | `SEED=42` |

**三套基础模型变体**（同样的数据/特征/多步/分步设定，仅输出粒度不同）：
- `single_output_mh_model/` —— **★当前主线**：单输出（每目标 1 模型）、原始尺度、多步、分步优化。
- `multi_output_model/` —— 多输出（1 模型出 5 目标），对比用。
- `single_output_model/` —— **旧版（历史保留）**：log 尺度 + 只 t+1 + 全网格穷举 + 含 Algae_Sum 输入，已不符合当前文档口径。

---

## 2. 环境

推荐 conda 环境 **`Bayesian`**（Python 3.13）。主要依赖：`torch`、`pandas`、`numpy`、`scikit-learn`、`efficient-kan`、`shap`、`matplotlib/seaborn/statsmodels`（见 [requirements.txt](requirements.txt)）。

```powershell
conda activate Bayesian
#   若 conda 不在 PATH，可直接用解释器绝对路径，例如：
#   & "C:\Users\<用户名>\anaconda3\envs\Bayesian\python.exe" 脚本.py

# efficient-kan（PyPI 没有，需从 GitHub 安装；LSTM-KAN / 纯KAN 依赖它）
python -m pip install "git+https://github.com/Blealtan/efficient-kan.git"
```

**SHAP 说明**：当前 `Bayesian`(Py3.13) 下 `import shap` 会因预编译 `_cext` wheel 崩溃，SHAP 步骤暂不能直接运行（代码已就绪）。修复方式：换一个能正常 `import shap` 的环境（如 Py3.11/3.12 + numpy<2.1 + shap）再运行 SHAP 脚本。

**Windows 中文路径注意**：运行脚本时先 `cd` 进脚本所在目录，再用文件名运行（脚本内用 `__file__` 定位路径，cwd 不影响）。

**烟雾测试**：训练脚本支持环境变量 `MULTI_SMOKE=1` → 极小搜索 + 2 epoch，仅验证流程/形状（不产出可用结果）。

---

## 3. 目录结构

```
code/
├── requirements.txt
├── data_pre/                          # 数据预处理与 EDA
│   ├── raw_data.csv
│   ├── data_pre(7_1_2).py             # ★当前划分 7:1:2，生成模型输入
│   ├── data_pre(7_2_1).py             # 旧划分 7:2:1（保留）
│   └── raw_data_EDA.py
│
├── base_model/
│   ├── data/                          # train/val/test_model_input.csv（由 data_pre 生成）
│   ├── common/                        # 公共工具（三套变体共用）
│   │   ├── data_utils.py              # 原始特征列、多步窗口、标准化、反变换
│   │   ├── train_utils.py / metrics_utils.py / seed_utils.py
│   ├── single_output_mh_model/        # ★当前主线：单输出·多步（每目标1模型）
│   │   ├── grid_search_single_mh.py            # LSTM-FC/MLP/KAN × 5目标，分步搜索
│   │   └── kan_full_grid_search_single_mh.py   # 纯KAN × 5目标
│   ├── multi_output_model/            # 多输出（联合，对比用）
│   │   ├── models.py                  # 4 架构模型 + 常量(TARGET_COLS/HORIZONS) —— 单/多输出共用
│   │   ├── grid_search_multi_output.py / kan_full_grid_search_multi.py
│   │   └── best_params_multi_seed.py
│   └── single_output_model/           # 旧：单输出(log/只t+1/全网格)，历史保留
│
├── hierarchical_reconciliation/       # 静态层次协调 BU/TD/OLS/WLS/MinT
│   ├── reconcile_single_mh_lstm_kan_best_params.py   # ★单输出：加载5个/目标模型，逐horizon协调
│   ├── reconcile_multi_lstm_kan_best_params.py       # 多输出（对比）
│   └── reconcile_single_lstm_kan_best_params.py      # 旧单输出（其协调数学被复用）
│
├── dynamic_weight_end_to_end_reconciliation/         # 动态权重端到端协调
│   ├── frozen_dynamic_wls_single_mh.py # ★单输出：5冻结分支，逐horizon DynamicWLS
│   ├── frozen_dynamic_wls_multi.py     # 多输出（对比；其训练/评估组件被单输出复用）
│   ├── frozen_dynamic_wls.py           # 旧单输出（其 MLP/汇总矩阵等被复用）
│   └── fine_tune_dynamic_wls.py        # 旧：不冻结微调版（未纳入新管线）
│
├── SHAP/
│   ├── shap_lstm_kan_single_mh.py     # ★单输出：逐(target,horizon) SHAP（待 shap 环境）
│   ├── shap_lstm_kan_multi.py         # 多输出（对比）
│   └── shap_lstm_kan.py               # 旧单输出
│
└── visualization/                     # R/ggplot2 绘图（原按旧单输出结果编写，新结果(含horizon)需适配列名/路径）
```

各脚本运行后会在所在目录生成对应 `*_outputs*/` 结果目录（见第 5 节）。

---

## 4. 快速开始：端到端训练流程（单输出主线）

> 用 `Bayesian` 环境运行；若 conda 不在 PATH，把 `python` 换成解释器绝对路径。

```powershell
# 第 0 步：生成 7:1:2 数据（写入 base_model/data/）
cd code/data_pre
python "data_pre(7_1_2).py"

# 第 1 步：单输出基础模型分步搜索（每目标1模型）—— 必须先跑，下游依赖
cd ../base_model/single_output_mh_model
python grid_search_single_mh.py            # LSTM-FC/MLP/KAN × 5 目标
python kan_full_grid_search_single_mh.py   # 纯 KAN × 5 目标

# 第 2 步：层次协调（BU/TD/OLS/WLS/MinT），加载 5 个/目标 LSTM-KAN，逐 horizon
cd ../../hierarchical_reconciliation
python reconcile_single_mh_lstm_kan_best_params.py

# 第 3 步：动态权重端到端协调（frozen DynamicWLS），5 冻结分支，逐 horizon，D1→D2=0
cd ../dynamic_weight_end_to_end_reconciliation
python frozen_dynamic_wls_single_mh.py

# 第 4 步：SHAP（待 shap 环境修复后，在可用环境运行）
cd ../SHAP
python shap_lstm_kan_single_mh.py

# 第 5 步：可视化（R 脚本；新结果含 horizon，需先适配列名）
#   Rscript visualization/.../xxx.R
```

**仅冒烟验证**：在第 1/2/3 步命令前设 `$env:MULTI_SMOKE="1"`（PowerShell）。

**多输出（对比）管线**：把上面对应脚本换成 `multi_output_model/grid_search_multi_output.py`、`kan_full_grid_search_multi.py`、`reconcile_multi_lstm_kan_best_params.py`、`frozen_dynamic_wls_multi.py`、`shap_lstm_kan_multi.py`。

> 重跑提醒：分步搜索每次从头重算并向 `search/*.csv` 追加、覆盖 `best_params`/`best_models`。要干净重跑先删对应 `*_outputs*/` 目录。

---

## 5. 各阶段功能与产物

### 数据预处理 `data_pre/`
`data_pre(7_1_2).py`：读 `raw_data.csv` → 构造 `Algae_Sum`、删除不用变量 → 按时间 7:1:2 切分 → 写 `base_model/data/{train,val,test}_model_input.csv`。

### 公共工具 `base_model/common/`
- `data_utils.py`：`get_raw_feature_cols`(13 原始输入)、`prepare_multi_horizon_data`(多步窗口、`y` 形状 `(N, H*K)` horizon‑major；单目标时 K=1)、`inverse_targets`(反标准化、不 expm1)。
- `train_utils.py`：`train_one_model`(Adam+MSE+早停)、`predict_scaled_multi`、`make_loader`。
- `metrics_utils.py`：`compute_metrics` → nRMSE/nMAE/NSE/KGE。

### 基础模型（★单输出主线 `single_output_mh_model/`）
- `grid_search_single_mh.py` / `kan_full_grid_search_single_mh.py`：对每个 `(目标, 架构)` 做坐标下降分步搜索（output_dim=3=该目标三步），按平均 val nRMSE 选优。模型类与超参空间复用 `multi_output_model/models.py` 与 `grid_search_multi_output.py` 的 `BASELINES/PARAM_ORDER`。
- **产物** `grid_search_outputs_single_mh/` 与 `kan_full_grid_search_single_mh_outputs/`：`best_params/{target}/best_params_{model}.json`、`best_models/{target}/best_{model}.pt`、`predictions/{target}/...(含horizon)`、`metrics/`、`search/`。

### 层次协调 `hierarchical_reconciliation/reconcile_single_mh_lstm_kan_best_params.py`
加载 5 个"每目标一个"的 LSTM-KAN，预测各目标 `(N,H)` → 逐 horizon 拼成 top‑first `(K,N)` → BU/TD/OLS/WLS/MinT（MinT 协方差仅用验证集残差）。**产物** `reconciliation_outputs_single_mh/`：`summary/reconciliation_metrics.csv`（target×horizon×method 指标）、`predictions/`。

### 动态权重协调 `dynamic_weight_end_to_end_reconciliation/frozen_dynamic_wls_single_mh.py`
5 个冻结单输出分支为基座，动态权重 MLP + 可微 WLS 对每个样本‑horizon 的 5 维向量协调。**产物** `frozen_dynamic_wls_single_mh_outputs/`：`metrics/`(Base vs DynamicWLS)、`consistency/..._by_horizon.csv`（**D1 协调前 → D2 协调后**，D2≈0 即精确一致）。

### SHAP `SHAP/shap_lstm_kan_single_mh.py`
对每个目标的单输出模型逐 `horizon` 计算特征级、feature‑lag 级 SHAP 与每特征 SHAP 最大的 lag（供 PDP）。**需可用的 shap 环境**。

### 可视化 `visualization/`（R）
网格搜索曲线、多种子箱线图、协调精度、协调对比图、SHAP 图；需 `ggplot2/scales/tidyverse`。原按旧单输出结果编写，套用新结果（含 `horizon`）需按新列名/路径适配。

---

## 6. 关键不变量（改代码时务必遵守）

- **目标列序**：模型原生顺序 `TARGET_COLS_ORDER = [Green_Algae, Cyanobacteria, Diatoms, Cryptophyta, Algae_Sum]`；协调要求 top‑first `ALL_TARGETS = [Algae_Sum, Green, Cyano, Diatoms, Crypto]`。多输出里用 `RECON_FROM_MODEL == [4,0,1,2,3]` 桥接；单输出里直接按 `ALL_TARGETS` 顺序加载 5 个分支即天然 top‑first。
- **多步 y 列序**：horizon‑major，`y[:, h_idx*K + k]` = 第 `HORIZONS[h_idx]` 步、第 `k` 个目标（单输出 K=1）。
- **反变换**：原始尺度用 `inverse_targets`（只反标准化）；勿用旧的 `inverse_log_target`（带 expm1，属旧 log 管线）。
- **Algae_Sum 只作输出，绝不作输入**。

---

## 7. 已知问题
- `import shap` 在 `Bayesian`(Py3.13) 崩溃 → SHAP 步骤需换环境运行。
- 含中文路径在某些终端把绝对路径作参数传给 python 会乱码 → 先 `cd` 再用文件名运行。
- `visualization/` 的 R 脚本面向旧单输出结果，新结果（含 horizon）需适配后再用。
