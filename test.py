import cv2
import numpy as np
import math
import os
from cvzone.HandTrackingModule import HandDetector

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as _DepthwiseConv2D

# ——— CUSTOM DEPTHWISE THAT DROPS `groups` ———
class DepthwiseConv2D(_DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config.pop("groups", None)
        return super().from_config(config)

def _infer_labels_from_data(data_dir: str) -> list[str]:
    if not os.path.isdir(data_dir):
        return []
    return sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )

def _load_labels(labels_path: str, data_dir: str) -> list[str]:
    if os.path.isfile(labels_path):
        labels: list[str] = []
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    labels.append(" ".join(parts[1:]))
                elif len(parts) == 1:
                    labels.append(parts[0])
        if labels:
            return labels
    labels = _infer_labels_from_data(data_dir)
    if labels:
        return labels
    raise FileNotFoundError(
        f"Could not load labels. Provide labels.txt at '{labels_path}' or ensure '{data_dir}' exists."
    )

# Load the trained Keras model and labels (repo-relative by default)
DATA_DIR = os.getenv("DATA_DIR", "Data")
model_path = os.getenv("MODEL_FILE", "keras_model_custom2.h5")
labels_path = os.getenv("LABELS_FILE", "labels.txt")

print("Loading Model...")
model = load_model(
    model_path,
    custom_objects={"DepthwiseConv2D": DepthwiseConv2D}
)
print("Model Loaded Successfully!")

# Load class labels
labels = _load_labels(labels_path, DATA_DIR)
print("Labels Loaded:", labels)

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open Camera.")
    exit()

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 224

try:
    print("Starting Hand Detection...")
    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture image.")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Crop the hand region
            y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
            x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size > 0:
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hGap + imgResize.shape[0], :] = imgResize

                # Normalize and prepare input for model
                imgWhite = imgWhite.astype(np.float32) / 255.0
                imgWhite = np.expand_dims(imgWhite, axis=0)

                # Predict gesture
                prediction = model.predict(imgWhite)
                classIndex = np.argmax(prediction)
                confidence = prediction[0][classIndex]
                label = f"{labels[classIndex]} ({confidence:.2f})"

                # Display label
                cv2.putText(
                    imgOutput, label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2
                )

        cv2.imshow('Image', imgOutput)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting on user request.")
            break

except KeyboardInterrupt:
    print("Program interrupted. Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and all windows closed.")
