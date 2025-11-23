"""
Data loader utilities for the FER image-folder dataset.

This module relies on torchvision.datasets.ImageFolder with the expected
directory layout:
data/
    archive/
        train/
            angry/
            ...
        test/
            angry/
            ...
"""

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _default_data_dir() -> Path:
    """Return the default data directory (project_root/data/archive)."""
    return Path(__file__).resolve().parents[1] / "data" / "archive"


def get_dataloaders(
    data_dir: Path | str | None = None,
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create train and test dataloaders using ImageFolder.

    Args:
        data_dir: Root directory containing "train" and "test" subfolders.
        batch_size: Batch size for both loaders.
        num_workers: Workers for background data loading.

    Returns:
        train_loader, test_loader, class_names
    """
    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    train_dir = root / "train"
    test_dir = root / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected train/test directories at {train_dir} and {test_dir}. "
            "Ensure the dataset is placed correctly."
        )

    common_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_dataset = datasets.ImageFolder(root=train_dir, transform=common_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=common_transforms)

    class_names = train_dataset.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader, class_names


__all__ = ["get_dataloaders"]
