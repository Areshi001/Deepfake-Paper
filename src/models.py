"""
Model builders for the five experimental configurations.
"""

import torch
import torch.nn as nn
import timm
from timm.models import mobilenetv3_small_100


# ---------------------------------------------------------------------------
# Full-face baseline — MobileNetV3 Small
# ---------------------------------------------------------------------------
def build_mobilenetv3(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """MobileNetV3-Small fine-tuned for binary classification."""
    model = timm.create_model(
        "mobilenetv3_small_100", pretrained=pretrained, num_classes=num_classes
    )
    return model


# ---------------------------------------------------------------------------
# Region-specific — EfficientNet-B0
# ---------------------------------------------------------------------------
def build_efficientnet_b0(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 fine-tuned for binary classification."""
    model = timm.create_model(
        "efficientnet_b0", pretrained=pretrained, num_classes=num_classes
    )
    return model


# ---------------------------------------------------------------------------
# Temporal — MobileNetV3 feature extractor + GRU
# ---------------------------------------------------------------------------
class TemporalGRU(nn.Module):
    """
    MobileNetV3-Small frozen feature extractor → GRU temporal model.

    Architecture
    ------------
    MobileNetV3-Small (frozen)
        ↓  1024-d frame feature
    8-frame sequence
        ↓
    GRU (hidden_size=128)
        ↓
    Binary classification head
    """

    def __init__(
        self,
        seq_len: int = 8,
        hidden_size: int = 128,
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        backbone = timm.create_model(
            "mobilenetv3_small_100", pretrained=pretrained, num_classes=0
        )
        # Feature dimension for mobilenetv3_small_100 is 576
        feat_dim = backbone.num_features
        self.backbone = backbone
        self.backbone.eval()  # frozen

        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch of frames (frozen backbone)."""
        with torch.no_grad():
            feat = self.backbone(x)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape ``(batch, seq_len, C, H, W)`` — a sequence of frames.

        Returns
        -------
        Tensor
            Logits of shape ``(batch, num_classes)``.
        """
        B, S, C, H, W = x.shape
        # Flatten batch and sequence dimensions
        x_flat = x.view(B * S, C, H, W)
        feats = self._extract_features(x_flat)  # (B*S, feat_dim)
        feats = feats.view(B, S, -1)  # (B, S, feat_dim)

        gru_out, _ = self.gru(feats)  # (B, S, hidden)
        last_hidden = gru_out[:, -1, :]  # (B, hidden)
        logits = self.head(last_hidden)  # (B, num_classes)
        return logits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
MODEL_BUILDERS = {
    "mobilenetv3": build_mobilenetv3,
    "efficientnet_b0": build_efficientnet_b0,
    "gru": lambda **kw: TemporalGRU(**kw),
}


def build_model(name: str, **kwargs) -> nn.Module:
    """Build a model by name."""
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](**kwargs)
