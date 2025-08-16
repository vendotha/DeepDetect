from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the new model and processor
processor = AutoImageProcessor.from_pretrained("HrutikAdsare/deepfake-detector-faceforensics")
model = AutoModelForImageClassification.from_pretrained("HrutikAdsare/deepfake-detector-faceforensics").to(device)
model.eval()

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))  # Optional: resize for performance
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
        labels = model.config.id2label
        return labels[prediction.item()], confidence.item()

from django.db import models
import numpy as np
import cv2

def predict_video(video_path, frame_skip=10, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((224, 224))
            inputs = processor(images=img, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                confidence, prediction = torch.max(probs, dim=1)
                predictions.append((model.config.id2label[prediction.item()], confidence.item()))

            if len(predictions) >= max_frames:
                break

        frame_count += 1

    cap.release()

    # Voting
    real_count = sum(1 for p, _ in predictions if "real" in p.lower())
    fake_count = sum(1 for p, _ in predictions if "fake" in p.lower())
    total = len(predictions)

    if fake_count > real_count:
        label = "Deepfake"
        confidence = fake_count / total
    else:
        label = "Real"
        confidence = real_count / total

    return label, round(confidence, 2)

# Create your models here.

