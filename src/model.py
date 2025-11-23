"""
Model definitions for facial emotion recognition.
"""

from pathlib import Path
from typing import Optional

import torch
from torch import nn


class EmotionCNN(nn.Module):
    """A compact CNN tailored for 48x48 grayscale facial emotion inputs."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )
        # With four 2x2 pools, 48x48 -> 3x3 spatial dims.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(x))
        return logits


def create_model(num_classes: int = 7, dropout: float = 0.3, device: Optional[torch.device] = None) -> EmotionCNN:
    """
    Factory to create the CNN and place it on the desired device.
    """
    model = EmotionCNN(num_classes=num_classes, dropout=dropout)
    if device is not None:
        model = model.to(device)
    return model


def load_model(
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
    num_classes: int = 7,
    dropout: float = 0.3,
) -> EmotionCNN:
    """
    Load a model from disk.

    Args:
        checkpoint_path: Path to .pth checkpoint containing state_dict.
        device: Torch device for loading.
        num_classes: Number of output classes.
        dropout: Dropout rate used at training time.
    """
    model = create_model(num_classes=num_classes, dropout=dropout, device=device)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device or "cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


__all__ = ["EmotionCNN", "create_model", "load_model"]
