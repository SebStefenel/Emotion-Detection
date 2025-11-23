"""
Real-time webcam emotion detection using OpenCV and the trained model.
"""

import cv2
import numpy as np
import torch
from pathlib import Path

from model import load_model

emotion_map = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def preprocess_face(gray_frame: np.ndarray, face_rect) -> torch.Tensor:
    x, y, w, h = face_rect
    face = gray_frame[y : y + h, x : x + w]
    face_resized = cv2.resize(face, (48, 48))
    face_normalized = face_resized.astype("float32") / 255.0
    face_normalized = (face_normalized - 0.5) / 0.5  # match training normalization
    tensor = torch.from_numpy(face_normalized).unsqueeze(0).unsqueeze(0)  # shape [1,1,48,48]
    return tensor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "saved_models" / "best_model.pth"
    model = load_model(model_path, device=device, num_classes=len(emotion_map))
    model.to(device)
    model.eval()

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam.")

    print("Press 'q' to quit.")
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_tensor = preprocess_face(gray, (x, y, w, h)).to(device)
                logits = model(face_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                label = emotion_map[pred_idx]
                confidence = probs[pred_idx]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"{label} ({confidence:.2f})"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
