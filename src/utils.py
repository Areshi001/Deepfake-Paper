"""
Shared utilities: dataset class, evaluation, device setup, and output directories.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset
from torch.amp import autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
BASE_DIR = Path("outputs")
DIRS = {
    "frames": BASE_DIR / "frames",
    "splits": BASE_DIR / "splits",
    "models": BASE_DIR / "models",
    "metrics": BASE_DIR / "metrics",
    "regions": BASE_DIR / "regions",
    "figures": BASE_DIR / "figures",
    "sample_predictions": BASE_DIR / "sample_predictions",
}


def ensure_dirs():
    """Create all required output directories."""
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FrameDataset(Dataset):
    """Generic frame-level dataset that reads images from disk."""

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = cv2.imread(row["path"])
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader) -> dict:
    """Evaluate a classifier and return accuracy, precision, recall, and AUC."""
    model.eval()
    preds, probs, targets = [], [], []
    for x, y in loader:
        x = x.to(DEVICE)
        with autocast(DEVICE.type if DEVICE.type == "cuda" else "cpu"):
            out = torch.softmax(model(x), dim=1)
        preds += out.argmax(1).cpu().tolist()
        probs += out[:, 1].cpu().tolist()
        targets += y.tolist()

    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        auc = 0.5

    return {
        "accuracy": round(accuracy_score(targets, preds), 4),
        "precision": round(precision_score(targets, preds, zero_division=0), 4),
        "recall": round(recall_score(targets, preds, zero_division=0), 4),
        "auc": round(float(auc), 4),
    }
