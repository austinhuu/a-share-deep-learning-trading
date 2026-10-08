from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .dataset import SequenceDataset


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float):
        improved = False

        if self.best_score is None:
            improved = True
        elif self.mode == "max" and score > self.best_score:
            improved = True
        elif self.mode == "min" and score < self.best_score:
            improved = True

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size: int = 64):
    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.BCELoss()

    losses = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            probs = model(X)
            loss = loss_fn(probs, y)

            losses.append(loss.item())
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_labels.extend(y.detach().cpu().numpy().tolist())

    avg_loss = float(np.mean(losses)) if losses else np.nan

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5

    return avg_loss, auc


def train_one_fold(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    lr: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 50,
    patience: int = 10,
    device: str = "cpu",
    weight_decay: float = 1e-4,
) -> Tuple[object, Dict]:
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=batch_size
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    loss_fn = nn.BCELoss()
    early_stopper = EarlyStopping(patience=patience, mode="max")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
    }

    best_state = deepcopy(model.state_dict())
    best_auc = -1.0

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            probs = model(X)
            loss = loss_fn(probs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss, val_auc = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = deepcopy(model.state_dict())

        early_stopper.step(val_auc)

        if early_stopper.should_stop:
            break

    model.load_state_dict(best_state)

    return model, history


def save_model(model, artifact_dir, filename: str = "model.pth"):
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), artifact_path / filename)