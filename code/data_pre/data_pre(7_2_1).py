import os
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 1. Path settings
# =========================
ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "data_pre" / "raw_data.csv"
OUTPUT_DIR = ROOT_DIR / "data_pre" / "data_pre_outputs(7_2_1)"
MODEL_DATA_DIR = ROOT_DIR / "base_model" / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. Preprocessing settings
# =========================
RENAME_COLUMNS = {
    "Bluegreen_Algae": "Cyanobacteria",
    "ATemperature": "AT",
    "WTemperature": "WT",
}

ALGAE_COLUMNS = [
    "Green_Algae",
    "Cyanobacteria",
    "Diatoms",
    "Cryptophyta",
]

DROP_COLUMNS = [
    "DO",
    "AT",
    "Conductivity",
    "Windspeed",
    "windspeed",
    "humidity",
    "NOx",
    "NOX",
    "NH4",
    "max_TEM",
    "min_TEM",
    "Algae_Total",
]

LOG1P_COLUMNS = [
    "Green_Algae",
    "Cyanobacteria",
    "Diatoms",
    "Cryptophyta",
    "Algae_Sum",
    "Turbidity",
    "DIP",
    "TP",
    "precipitation",
    "NPR",
]


def validate_columns(df, required_cols, step_name):
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{step_name} 缺少必要变量: {missing_cols}")


def add_log1p_columns(df, log_cols):
    df = df.copy()

    for col in log_cols:
        if col not in df.columns:
            continue

        min_value = df[col].min(skipna=True)
        if pd.notna(min_value) and min_value <= -1:
            raise ValueError(f"{col} 存在 <= -1 的值，不能进行 log1p 变换。")

        df[f"log_{col}"] = np.log1p(df[col])

    return df


def split_by_time(df, train_ratio=0.7, val_ratio=0.2):
    n_rows = len(df)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def save_split(df, split_name, filename):
    output_path = OUTPUT_DIR / filename
    model_path = MODEL_DATA_DIR / filename

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    df.to_csv(model_path, index=False, encoding="utf-8-sig")

    print(f"{split_name}: {df.shape} -> {output_path}")


# =========================
# 3. Load and basic cleaning
# =========================
df = pd.read_csv(INPUT_PATH)
validate_columns(df, ["date"], "读取原始数据")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df = df.rename(columns={old: new for old, new in RENAME_COLUMNS.items() if old in df.columns})

validate_columns(df, ALGAE_COLUMNS, "构造 Algae_Sum")
df["Algae_Sum"] = df[ALGAE_COLUMNS].sum(axis=1)

existing_drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
df = df.drop(columns=existing_drop_cols)

df = add_log1p_columns(df, LOG1P_COLUMNS)


# =========================
# 4. Train / validation / test split
# =========================
train_df, val_df, test_df = split_by_time(df, train_ratio=0.7, val_ratio=0.2)

split_info = pd.DataFrame(
    {
        "dataset": ["train", "val", "test"],
        "rows": [len(train_df), len(val_df), len(test_df)],
        "ratio": [len(train_df) / len(df), len(val_df) / len(df), len(test_df) / len(df)],
        "start_date": [train_df["date"].min(), val_df["date"].min(), test_df["date"].min()],
        "end_date": [train_df["date"].max(), val_df["date"].max(), test_df["date"].max()],
    }
)
split_info.to_csv(OUTPUT_DIR / "split_info.csv", index=False, encoding="utf-8-sig")


# =========================
# 5. Save model-ready datasets
# =========================
train_df.to_csv(OUTPUT_DIR / "train_model_input.csv", index=False, encoding="utf-8-sig")
val_df.to_csv(OUTPUT_DIR / "val_model_input.csv", index=False, encoding="utf-8-sig")
test_df.to_csv(OUTPUT_DIR / "test_model_input.csv", index=False, encoding="utf-8-sig")

# Files used directly by existing base_model scripts.
for prefix, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    split_df.to_csv(MODEL_DATA_DIR / f"{prefix}_Algae_Sum_input.csv", index=False, encoding="utf-8-sig")
    split_df.to_csv(MODEL_DATA_DIR / f"{prefix}_four_Algae_input.csv", index=False, encoding="utf-8-sig")

print("Data preprocessing completed.")
print(f"Raw data: {INPUT_PATH}")
print(f"Preprocessing outputs: {OUTPUT_DIR}")
print(f"Model-ready data: {MODEL_DATA_DIR}")
print(split_info)
