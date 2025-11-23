# Facial Emotion Recognition (PyTorch)

End-to-end facial emotion recognition using a FER-style dataset organized in `train`/`test` folders. The project includes training, evaluation, a real-time webcam demo, and a Streamlit web app.

## Project Structure
```
emotion-detection/
├── data/
│   └── archive/
│       ├── train/
│       │   ├── angry/ ...
│       └── test/
│           ├── angry/ ...
├── src/
│   ├── dataloader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── realtime.py
├── app/
│   └── app.py
├── saved_models/
│   └── best_model.pth   # created after training
├── results/
│   └── confusion_matrix.png (created by evaluate.py)
├── requirements.txt
└── README.md
```

## Setup
1. Ensure the dataset is placed at `data/archive/train` and `data/archive/test` with class subfolders (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`).
2. Create and activate a virtual environment (optional).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training
Run from the project root:
```bash
python src/train.py --epochs 25 --batch_size 64 --lr 1e-3 --num_workers 4
```
Optional: `--data_dir` to point to a custom dataset root and `--save_dir` to choose another output folder. The best weights are saved to `saved_models/best_model.pth`.

## Evaluation
```bash
python src/evaluate.py --num_workers 4
```
Optional flags: `--data_dir` for a custom dataset location and `--checkpoint` for a custom weights path. Outputs accuracy, precision/recall/F1, and saves `results/confusion_matrix.png`.

## Real-Time Webcam Demo
```bash
python src/realtime.py
```
Requires a webcam. Press `q` to quit.

## Streamlit App
```bash
streamlit run app/app.py
```
Upload an image to detect faces and view predicted emotions with probabilities.

## Notes
- Training uses grayscale 48×48 inputs normalized to mean 0.5 and std 0.5.
- GPU is used automatically when available.
- Replace `saved_models/best_model.pth` with your own trained weights if needed.

## Example Outputs
- Confusion matrix: `results/confusion_matrix.png` (generated after evaluation).
- Annotated images appear in the Streamlit UI; webcam overlays appear in the live feed.
