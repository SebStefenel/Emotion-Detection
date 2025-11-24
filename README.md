# Facial Emotion Recognition (PyTorch, OpenCV, Streamlit)

End-to-end facial emotion recognition: train a CNN on a folder-based FER dataset, evaluate it, and serve predictions via a webcam demo or Streamlit UI.

## Project description
- **Problem**: Detect a face in an image and classify it into one of seven emotions (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`).
- **Why it matters**: Emotion cues improve UX, safety (fatigue/stress monitoring), accessibility, and analytics.
- **How it works**: OpenCV finds faces, images are normalized to 48×48 grayscale, and a compact CNN (trained with PyTorch) outputs emotion probabilities. The saved weights live in `saved_models/best_model.pth` after training.

## Dataset
Folder-based layout (FER2013-style). Place data under `data/archive`:
```
data/archive/
  train/
    angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
  test/
    angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
```
Images are resized to 48×48 and normalized with mean=0.5, std=0.5. Classes are inferred from folder names.

## Model architecture
- 4 convolutional blocks (Conv-BN-ReLU-MaxPool + dropout) → spatial size 48×48 → 3×3.
- Classifier: Flatten → Linear(256*3*3 → 512) → ReLU + dropout → Linear → 7 logits.
- Loss/optim: CrossEntropyLoss + Adam. GPU is used automatically if available.

## Results
- Run `src/evaluate.py` after training to print a classification report and test accuracy; it also saves `results/confusion_matrix.png`.
- `saved_models/best_model.pth` stores the best checkpoint found during training (based on test accuracy). Update this section with your latest metrics after a run.

## Example predictions
- `app/app.py` (Streamlit): upload an image, faces are detected, and predicted emotions + probability bars are shown.
- `src/realtime.py`: live webcam demo; press `q` to quit. Both use the same trained CNN weights.

## Quick start
From the repo root:
```bash
# install deps (macOS/arm; adjust python path if needed)
/opt/homebrew/bin/python3.11 -m pip install --user --break-system-packages -r requirements.txt

# or install into a local target to avoid permission issues
/opt/homebrew/bin/python3.11 -m pip install --target "$(pwd)/.local-packages" -r requirements.txt

# train (CPU or GPU if available)
PYTHONPATH="$(pwd)/.local-packages:$PYTHONPATH" \
/opt/homebrew/bin/python3.11 src/train.py --epochs 25 --batch_size 64 --lr 1e-3 --num_workers 0

# evaluate (writes results/confusion_matrix.png)
PYTHONPATH="$(pwd)/.local-packages:$PYTHONPATH" \
/opt/homebrew/bin/python3.11 src/evaluate.py --data_dir data/archive --checkpoint saved_models/best_model.pth
```
If you saved the checkpoint elsewhere (e.g., `/tmp/best_model.pth`), point `--checkpoint` there or copy it into `saved_models/`.

## Streamlit
```bash
PYTHONPATH="$(pwd)/.local-packages:$PYTHONPATH" \
/opt/homebrew/bin/python3.11 -m streamlit run app/app.py
```
Upload an (upright) image to see detected faces, predicted emotions, and probability bars.

## Commands (common options)
- `src/train.py`: `--epochs`, `--batch_size`, `--lr`, `--num_workers`, `--data_dir`, `--save_dir`
- `src/evaluate.py`: `--data_dir`, `--checkpoint`, `--num_workers`
- `src/realtime.py`: runs webcam Haar-cascade detection + model inference; press `q` to quit.
- `app/app.py`: Streamlit UI for image upload; faces are auto-detected and labeled.

## Technologies used
- PyTorch + torchvision (CNN, training loop)
- OpenCV (Haar face detection, preprocessing)
- Streamlit (UI for uploads/visualization)
- scikit-learn + matplotlib (metrics, confusion matrix)
- tqdm, numpy, pandas (utilities)

## Troubleshooting
- If pip refuses to write to `~/Library/Python/...`, add `--user --break-system-packages` or use `--target "$(pwd)/.local-packages"`.
- If saving checkpoints under `saved_models/` fails on macOS, use `--save_dir /tmp` and copy `/tmp/best_model.pth` into `saved_models/`.
- For DataLoader shared-memory issues on macOS, set `--num_workers 0`.

## Project structure
```
src/
  dataloader.py   # ImageFolder loaders with grayscale/normalize transforms
  model.py        # CNN (4 conv blocks + FC classifier)
  train.py        # training loop with tqdm and best-model saving
  evaluate.py     # metrics + confusion matrix plot to results/
  realtime.py     # webcam demo with Haar cascade
app/
  app.py          # Streamlit app for uploads and predictions
saved_models/     # best_model.pth (created after training)
results/          # confusion_matrix.png (created after evaluation)
requirements.txt
```
