"""
Evaluation script for facial emotion recognition.
"""

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataloader import get_dataloaders
from model import load_model


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Confusion matrix saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained facial emotion recognition model.")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory containing archive/train and archive/test.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to saved model checkpoint.")
    parser.add_argument("--num_workers", type=int, default=2, help="Dataloader worker count.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir, batch_size=64, num_workers=args.num_workers
    )

    project_root = Path(__file__).resolve().parents[1]
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else project_root / "saved_models" / "best_model.pth"
    model = load_model(checkpoint_path, device=device, num_classes=len(class_names))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    results_dir = project_root / "results"
    plot_confusion_matrix(cm, class_names, results_dir / "confusion_matrix.png")

    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
