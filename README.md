# Facial Emotion Recognition (PyTorch, ImageFolder)

End‑to‑end facial emotion recognition using a folder‑based FER dataset. Includes training, evaluation, webcam demo, and a Streamlit UI.

## Dataset layout
Place data under `data/archive`:
```
data/archive/
  train/
    angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
  test/
    angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
```
Classes are inferred from folder names; images are resized to 48×48 grayscale.

## Quick start
From the repo root:
```bash
# install deps (macOS/arm; adjust python path if needed)
/opt/homebrew/bin/python3.11 -m pip install --user --break-system-packages -r requirements.txt

# train (CPU or GPU if available)
PYTHONPATH="$(pwd)/.local-packages:$PYTHONPATH" \
/opt/homebrew/bin/python3.11 src/train.py --epochs 25 --batch_size 64 --lr 1e-3 --num_workers 0

# evaluate (writes results/confusion_matrix.png)
PYTHONPATH="$(pwd)/.local-packages:$PYTHONPATH" \
/opt/homebrew/bin/python3.11 src/evaluate.py --data_dir data/archive --checkpoint saved_models/best_model.pth
```
If you saved the checkpoint to `/tmp/best_model.pth`, point `--checkpoint` there or copy it into `saved_models/`.

## Commands (common options)
- `src/train.py`: `--epochs`, `--batch_size`, `--lr`, `--num_workers`, `--data_dir`, `--save_dir`
- `src/evaluate.py`: `--data_dir`, `--checkpoint`, `--num_workers`
- `src/realtime.py`: runs webcam Haar-cascade detection + model inference; press `q` to quit.
- `app/app.py`: Streamlit UI for image upload; faces are auto-detected and labeled.

## Streamlit
```bash
PYTHONPATH="$(pwd)/.local-packages:$PYTHONPATH" \
/opt/homebrew/bin/python3.11 -m streamlit run app/app.py
```
Upload an image to see detected faces, predicted emotions, and probability bars.

## Troubleshooting
- If pip refuses to write to `~/Library/Python/...`, add `--user --break-system-packages` (as above) or install to a local target: `--target "$(pwd)/.local-packages"`.
- If saving checkpoints under `saved_models/` fails on macOS, use `--save_dir /tmp` and then copy `/tmp/best_model.pth` into `saved_models/`.
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

## Notes
- Normalization: mean=0.5, std=0.5 on grayscale 48×48 images.
- Uses Adam + CrossEntropyLoss; training defaults to 25 epochs, batch size 64.
- GPU is used automatically if available.***
