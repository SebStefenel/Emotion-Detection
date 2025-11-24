"""
Training script for the facial emotion recognition model.
"""

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import get_dataloaders
from model import create_model


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    progress = tqdm(loader, desc="Train", leave=False)
    for inputs, labels in progress:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        acc = accuracy_from_logits(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_acc += acc * inputs.size(0)
        progress.set_postfix(loss=loss.item(), acc=acc)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = accuracy_from_logits(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_acc += acc * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train facial emotion recognition model.")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory containing archive/train and archive/test.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--num_workers", type=int, default=2, help="Dataloader worker count.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save best model.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = create_model(num_classes=len(class_names), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    project_root = Path(__file__).resolve().parents[1]
    save_dir = Path(args.save_dir) if args.save_dir else project_root / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            # Save via a temporary file (in system temp dir) to avoid macOS copy/rename issues.
            import tempfile, shutil, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
                tmp_path = Path(tmp_file.name)
            torch.save(model.state_dict(), tmp_path)
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            if best_model_path.exists():
                try:
                    best_model_path.unlink()
                except Exception:
                    pass
            shutil.copyfile(tmp_path, best_model_path)
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            try:
                os.chmod(best_model_path, 0o644)
            except Exception:
                pass

            print(f"Saved new best model to {best_model_path} (acc={best_acc:.4f})")

    print(f"Training complete. Best test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
