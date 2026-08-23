"""
Evaluate a trained model checkpoint.

Usage
-----
    python src/evaluate.py --checkpoint outputs/models/mobilenetv3_best.pth --config configs/baseline.yaml
"""

import argparse
from pathlib import Path

import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.utils import DEVICE, DIRS, FrameDataset, evaluate
from src.preprocessing import get_val_transform
from src.models import build_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_checkpoint(checkpoint_path: str, config_path: str):
    cfg = load_config(config_path)
    model_name = cfg["model"]

    region = cfg.get("region")
    csv_path = Path(cfg.get("frames_csv", DIRS["splits"] / "frames_split.csv"))
    if region:
        csv_path = DIRS["metrics"] / "region_frames.csv"
    df = pd.read_csv(csv_path)

    if region:
        df = df[df["region"] == region].reset_index(drop=True)

    te_df = df[df["split"] == "test"]
    tf_val = get_val_transform()
    te_ld = DataLoader(
        FrameDataset(te_df, tf_val),
        batch_size=cfg.get("batch_size", 64),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model_kwargs = cfg.get("model_kwargs", {})
    model = build_model(model_name, **model_kwargs).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    metrics = evaluate(model, te_ld)
    print("Evaluation metrics:", metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.config)
