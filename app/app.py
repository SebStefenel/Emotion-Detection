import cv2
import numpy as np
import streamlit as st
import torch
from pathlib import Path
from PIL import Image
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.model import load_model

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def load_cascade() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    classifier = cv2.CascadeClassifier(str(cascade_path))
    if classifier.empty():
        raise RuntimeError("Unable to load Haar cascade.")
    return classifier


def preprocess_face(face_img: np.ndarray) -> torch.Tensor:
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (48, 48))
    face_norm = face_resized.astype("float32") / 255.0
    face_norm = (face_norm - 0.5) / 0.5
    tensor = torch.from_numpy(face_norm).unsqueeze(0).unsqueeze(0)
    return tensor


@st.cache_resource
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "saved_models" / "best_model.pth"
    model = load_model(model_path, device=device, num_classes=len(EMOTION_LABELS))
    model.to(device)
    model.eval()
    return model, device


def predict_emotions(image_bgr: np.ndarray, face_cascade: cv2.CascadeClassifier, model, device):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    results = []
    for (x, y, w, h) in faces:
        face_img = image_bgr[y : y + h, x : x + w]
        tensor = preprocess_face(face_img).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        results.append(
            {
                "bbox": (x, y, w, h),
                "label": EMOTION_LABELS[pred_idx],
                "confidence": probs[pred_idx],
                "probs": probs,
            }
        )
    return faces, results


def draw_results(image_bgr: np.ndarray, results):
    for res in results:
        x, y, w, h = res["bbox"]
        label = res["label"]
        confidence = res["confidence"]
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image_bgr,
            f"{label}: {confidence:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return image_bgr


def main():
    st.set_page_config(page_title="Facial Emotion Recognition", layout="wide")
    st.title("Facial Emotion Recognition")
    st.write("Upload an image to detect faces and predict emotions.")

    model, device = load_trained_model()
    face_cascade = load_cascade()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        faces, results = predict_emotions(image_bgr, face_cascade, model, device)
        annotated = draw_results(image_bgr.copy(), results)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Detected Faces")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

        with col2:
            st.subheader("Predictions")
            if not results:
                st.write("No faces detected.")
            for idx, res in enumerate(results, start=1):
                st.write(f"Face {idx}: **{res['label']}** ({res['confidence']:.2f})")
                prob_data = {label: float(prob) for label, prob in zip(EMOTION_LABELS, res["probs"])}
                st.bar_chart(prob_data)


if __name__ == "__main__":
    main()
