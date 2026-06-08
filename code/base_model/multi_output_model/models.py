"""
多输出基础模型定义（一次性输出全部目标）。

与 single_output_model 的区别仅在于输出维度由 1 改为 output_dim（=目标数 K），
其余结构、超参数含义完全一致，便于单输出 vs 多输出公平对比。

规范目标顺序 TARGET_COLS_ORDER 是多输出 y 矩阵列序 / 输出头顺序的唯一真源，
下游需要 top-first 顺序时按名字派生索引，不要在别处手写下标。
"""

from __future__ import annotations

from torch import nn

try:
    from efficient_kan import KAN
except ImportError:  # pragma: no cover - 缺依赖时延迟到实例化再报错
    KAN = None


# 模型原生顺序：四个底层藻 + 总藻（与研究进展文档“模型输出”一致）。
TARGET_COLS_ORDER = [
    "Green_Algae",
    "Cyanobacteria",
    "Diatoms",
    "Cryptophyta",
    "Algae_Sum",
]

# 原始尺度目标列名（不做 log，符合研究进展文档）。训练 y、y_scaler 直接用这些原始列。
TARGET_COLS = list(TARGET_COLS_ORDER)

# 目标名 -> 原始数据列名（原始尺度下二者同名，恒等映射；供单输出按目标取列用）。
TARGET_MAP_RAW = {name: name for name in TARGET_COLS_ORDER}
# 目标名 -> log1p 数据列名（log 尺度）。
TARGET_MAP_LOG = {name: f"log_{name}" for name in TARGET_COLS_ORDER}

# ============================================================
# 尺度开关（方案A）：USE_LOG=True 用 log1p 尺度（log 输入特征 + log 目标 + expm1 反变换）；
# False 用原始尺度。各管线据此选择 feature 列函数、目标列、反变换函数（及 DynamicWLS 是否 expm1）。
# 切换 raw<->log 只需改这一个开关并重跑。
# ============================================================
USE_LOG = True

# 训练用目标列与“目标名->训练列”映射，随 USE_LOG 切换。
TARGET_MAP_TRAIN = TARGET_MAP_LOG if USE_LOG else TARGET_MAP_RAW
TARGET_TRAIN_COLS = [TARGET_MAP_TRAIN[name] for name in TARGET_COLS_ORDER]

# 输入特征是否也用 log（独立于目标的 USE_LOG）。
# True: 用 get_log_feature_cols（对偏态输入做 log1p）；False: get_raw_feature_cols（输入全原始）。
LOG_INPUTS = True

# 超参数搜索方式："coordinate"=分步坐标下降（当前）；"grid"=小范围网格穷举。
SEARCH_METHOD = "coordinate"

# SHAP 去自身特征用：原始尺度下目标的“自身特征”就是同名原始列。
# 四个底层藻同名列在输入中（会被剔除）；Algae_Sum 不在输入特征中，匹配不到、无影响。
SELF_FEATURE_MAP = {name: name for name in TARGET_COLS_ORDER}

# 多步预测：一次性输出 t+1, t+2, t+3。模型输出维度 = len(HORIZONS) * len(TARGET_COLS)。
HORIZONS = [1, 2, 3]

MODEL_DISPLAY = {
    "LSTM_FC": "LSTM-FC",
    "LSTM_MLP": "LSTM-MLP",
    "LSTM_KAN": "LSTM-KAN",
    "KAN": "KAN",
}
MODEL_STEM = {
    "LSTM_FC": "lstm_fc",
    "LSTM_MLP": "lstm_mlp",
    "LSTM_KAN": "lstm_kan",
    "KAN": "kan",
}


class LSTMFC(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, output_dim: int):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        hidden = self.dropout(out[:, -1, :])
        return self.fc(hidden)


class LSTMMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        mlp_hidden_dim: int,
        mlp_num_layers: int,
        output_dim: int,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        layers = []
        in_dim = hidden_dim
        for _ in range(mlp_num_layers):
            layers.extend([nn.Linear(in_dim, mlp_hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = mlp_hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.mlp(out[:, -1, :])


class LSTMKAN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        kan_hidden_dim: int | None,
        grid_size: int,
        spline_order: int,
        output_dim: int,
    ):
        super().__init__()
        if KAN is None:
            raise ImportError("未检测到 efficient-kan，请先安装 efficient-kan 后再运行 LSTM-KAN。")

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        layers_hidden = (
            [hidden_dim, output_dim]
            if kan_hidden_dim is None
            else [hidden_dim, kan_hidden_dim, output_dim]
        )
        self.kan = KAN(layers_hidden, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x):
        out, _ = self.lstm(x)
        hidden = self.dropout(out[:, -1, :])
        return self.kan(hidden)


class KANRegressor(nn.Module):
    """纯 KAN：把 (N, lookback, n_features) 展平为 (N, lookback*n_features) 后送入 KAN。"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kan_hidden_dim: int = 64,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if KAN is None:
            raise ImportError("未检测到 efficient-kan，请先安装 efficient-kan 后再运行 KAN。")

        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        layers_hidden = [input_dim, kan_hidden_dim, output_dim]
        self.kan = KAN(layers_hidden, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        return self.kan(x)


def build_model(model_name: str, input_dim: int, params: dict, output_dim: int) -> nn.Module:
    """
    构造多输出模型。注意 input_dim 含义随架构不同：
    - LSTM 家族：input_dim = 特征数（n_features）；
    - 纯 KAN：input_dim = lookback * n_features（展平后）。
    调用方负责按架构传入正确的 input_dim。
    """
    if model_name == "LSTM_FC":
        return LSTMFC(input_dim, params["hidden_dim"], params["num_layers"], params["dropout"], output_dim)
    if model_name == "LSTM_MLP":
        return LSTMMLP(
            input_dim,
            params["hidden_dim"],
            params["num_layers"],
            params["dropout"],
            params["mlp_hidden_dim"],
            params["mlp_num_layers"],
            output_dim,
        )
    if model_name == "LSTM_KAN":
        return LSTMKAN(
            input_dim,
            params["hidden_dim"],
            params["num_layers"],
            params["dropout"],
            params["kan_hidden_dim"],
            params["grid_size"],
            params["spline_order"],
            output_dim,
        )
    if model_name == "KAN":
        return KANRegressor(
            input_dim,
            output_dim,
            params["kan_hidden_dim"],
            params["grid_size"],
            params["spline_order"],
            params["dropout"],
        )
    raise ValueError(f"未知模型: {model_name}")
