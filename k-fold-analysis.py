# kfold_evaluate_model.py

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as _DepthwiseConv2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# —— CONFIGURATION —— 
DATA_DIR    = os.getenv("DATA_DIR", "Data")
MODEL_FILE  = os.getenv("MODEL_FILE", "keras_model_custom2.h5")
LABELS_FILE = os.getenv("LABELS_FILE", "labels.txt")
IMG_SIZE         = (224, 224)
BATCH_SIZE       = 32
N_SPLITS         = 5                               # Number of folds

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

# —— CUSTOM DEPTHWISE LAYER FOR LOADING SAVED MODEL —— 
class DepthwiseConv2D(_DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config.pop("groups", None)
        return super().from_config(config)

# —— LOAD YOUR SAVED MODEL ONCE —— 
if not os.path.isfile(MODEL_FILE):
    raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
model = load_model(MODEL_FILE, custom_objects={"DepthwiseConv2D": DepthwiseConv2D})

# —— READ LABELS & PREPARE DATAFRAME —— 
labels = _load_labels(LABELS_FILE, DATA_DIR)

# Collect file paths and true labels
filepaths, true_labels = [], []
for lbl in labels:
    class_dir = os.path.join(DATA_DIR, lbl)
    if not os.path.isdir(class_dir):
        continue
    for fname in os.listdir(class_dir):
        path = os.path.join(class_dir, fname)
        if os.path.isfile(path):
            filepaths.append(path)
            true_labels.append(lbl)

# Encode labels to integers
le = LabelEncoder()
y = le.fit_transform(true_labels)
X = np.array(filepaths)

# —— SET UP STRATIFIED K-FOLD —— 
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

fold_accuracies = []

print("Evaluating saved model with Stratified K-Fold:")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    val_paths = X[val_idx]
    val_labels = y[val_idx]
    
    correct = 0
    for path, true_idx in zip(val_paths, val_labels):
        img = load_img(path, target_size=IMG_SIZE)
        x   = img_to_array(img).astype("float32") / 255.0
        x   = np.expand_dims(x, 0)
        preds = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        if pred_idx == true_idx:
            correct += 1
    
    acc = correct / len(val_paths)
    fold_accuracies.append(acc)
    print(f" Fold {fold}: {acc:.2%} accuracy on {len(val_paths)} samples")

# —— SUMMARY —— 
avg_acc = np.mean(fold_accuracies)
print(f"\nAverage accuracy across {N_SPLITS} folds: {avg_acc:.2%}")

# —— PLOT —— 
plt.figure()
plt.plot(range(1, N_SPLITS+1), fold_accuracies, marker='o')
plt.title("Saved Model Accuracy per Fold")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.show()
