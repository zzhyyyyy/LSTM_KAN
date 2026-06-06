from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _state_dict_to_cpu(state_dict: dict) -> dict:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    max_epochs: int,
    patience: int,
    device: torch.device,
    weight_decay: float = 0.0,
) -> tuple[list[dict], dict, int, float, float]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    train_loss_at_best = float("inf")
    wait = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_sum, train_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * xb.size(0)
            train_n += xb.size(0)
        train_loss = train_sum / max(train_n, 1)

        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                val_sum += loss.item() * xb.size(0)
                val_n += xb.size(0)
        val_loss = val_sum / max(val_n, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_epoch = epoch
            train_loss_at_best = train_loss
            best_state = _state_dict_to_cpu(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        best_state = _state_dict_to_cpu(model.state_dict())
    model.load_state_dict(copy.deepcopy(best_state))
    return history, best_state, best_epoch, train_loss_at_best, best_val_loss


@torch.no_grad()
def predict_scaled(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    dummy_y = np.zeros((len(x), 1), dtype=np.float32)
    loader = make_loader(x, dummy_y, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    preds = []
    for xb, _ in loader:
        preds.append(model(xb.to(device)).detach().cpu().numpy())
    return np.vstack(preds).reshape(-1, 1)


@torch.no_grad()
def predict_scaled_multi(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
    output_dim: int,
) -> np.ndarray:
    """多输出预测：返回 (N, output_dim) 标准化预测矩阵（不做单列 reshape）。"""
    dummy_y = np.zeros((len(x), output_dim), dtype=np.float32)
    loader = make_loader(x, dummy_y, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    preds = []
    for xb, _ in loader:
        preds.append(model(xb.to(device)).detach().cpu().numpy())
    return np.vstack(preds).reshape(-1, output_dim)


def save_loss_history(history: list[dict], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False, encoding="utf-8-sig")
