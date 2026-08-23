"""
Training script — supports all five model configurations via YAML configs.

Usage
-----
    python src/train.py --config configs/baseline.yaml
    python src/train.py --config configs/periocular.yaml
    python src/train.py --config configs/mouth.yaml
    python src/train.py --config configs/cheeks.yaml
    python src/train.py --config configs/gru.yaml
"""

import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.amp import autocast, GradScaler

from src.utils import DEVICE, DIRS, FrameDataset, evaluate
from src.preprocessing import get_train_transform, get_val_transform
from src.models import build_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_loader(df: pd.DataFrame, tf, batch_size: int, shuffle: bool = False):
    ds = FrameDataset(df, tf)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )


def train(config_path: str):
    cfg = load_config(config_path)

    model_name = cfg["model"]
    epochs = cfg.get("epochs", 10)
    lr = cfg.get("lr", 3e-4)
    weight_decay = cfg.get("weight_decay", 1e-4)
    batch_size = cfg.get("batch_size", 64)
    patience_limit = cfg.get("patience", 3)

    # ---- Data ----
    region = cfg.get("region")  # None for full-face
    csv_path = Path(cfg.get("frames_csv", DIRS["splits"] / "frames_split.csv"))
    if region:
        csv_path = DIRS["metrics"] / "region_frames.csv"
    df = pd.read_csv(csv_path)

    if region:
        df = df[df["region"] == region].reset_index(drop=True)

    tr_df = df[df["split"] == "train"]
    vl_df = df[df["split"] == "val"]
    te_df = df[df["split"] == "test"]

    tf_train = get_train_transform()
    tf_val = get_val_transform()

    tr_ld = make_loader(tr_df, tf_train, batch_size, shuffle=True)
    vl_ld = make_loader(vl_df, tf_val, batch_size)
    te_ld = make_loader(te_df, tf_val, batch_size)

    # ---- Model ----
    model_kwargs = cfg.get("model_kwargs", {})
    model = build_model(model_name, **model_kwargs).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, steps_per_epoch=len(tr_ld), epochs=epochs
    )
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler(DEVICE.type if DEVICE.type == "cuda" else "cpu")

    model_alias = cfg.get("model_alias", model_name)
    print(f"Training {model_alias}...")

    best_auc, patience = 0.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in tr_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with autocast(DEVICE.type if DEVICE.type == "cuda" else "cpu"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total_loss += loss.item()

        vm = evaluate(model, vl_ld)
        print(
            f"  Epoch {epoch}/{epochs} | loss={total_loss / len(tr_ld):.4f} "
            f"| val AUC={vm['auc']} | val acc={vm['accuracy']}"
        )

        if vm["auc"] > best_auc:
            best_auc = vm["auc"]
            torch.save(
                model.state_dict(), DIRS["models"] / f"{model_alias}_best.pth"
            )
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"  Early stop at epoch {epoch}")
                break

    # ---- Test ----
    model.load_state_dict(
        torch.load(DIRS["models"] / f"{model_alias}_best.pth", weights_only=True)
    )
    test_metrics = evaluate(model, te_ld)
    print(f"\nTest metrics ({model_alias}):", test_metrics)

    pd.DataFrame([test_metrics]).to_csv(
        DIRS["metrics"] / f"{model_alias}_results.csv", index=False
    )
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    train(args.config)
