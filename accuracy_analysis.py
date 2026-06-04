# accuracy_analysis.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as _DepthwiseConv2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 1. Configuration
DATA_DIR    = os.getenv("DATA_DIR", "Data")
LABELS_FILE = os.getenv("LABELS_FILE", "labels.txt")
MODEL_FILE  = os.getenv("MODEL_FILE", "keras_model_custom2.h5")

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

# 2. Custom layer definition for loading the model
class DepthwiseConv2D(_DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config.pop("groups", None)
        return super().from_config(config)

# 3. Load the trained model
model = load_model(MODEL_FILE, custom_objects={"DepthwiseConv2D": DepthwiseConv2D})

# 4. Read label names (expects "index LabelName" per line)
labels_txt = _load_labels(LABELS_FILE, DATA_DIR)

# 5. Gather all samples and their true labels
file_paths, file_labels = [], []
for label in labels_txt:
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_dir):
        continue
    for fname in os.listdir(class_dir):
        path = os.path.join(class_dir, fname)
        if os.path.isfile(path):
            file_paths.append(path)
            file_labels.append(label)

file_paths  = np.array(file_paths)
file_labels = np.array(file_labels)

# 6. Split into 80% train / 20% test (stratified)
train_paths, test_paths, train_lbls, test_lbls = train_test_split(
    file_paths, file_labels, test_size=0.2,
    stratify=file_labels, random_state=42
)

# 7. Evaluate overall and per-class accuracy
correct = 0
per_class_stats = {label: {"correct": 0, "total": 0} for label in labels_txt}

for path, true_label in zip(test_paths, test_lbls):
    img = load_img(path, target_size=(224, 224))
    x   = img_to_array(img).astype("float32") / 255.0
    x   = np.expand_dims(x, axis=0)
    preds = model.predict(x, verbose=0)
    pred_idx = np.argmax(preds[0])
    pred_label = labels_txt[pred_idx]
    
    per_class_stats[true_label]["total"] += 1
    if pred_label == true_label:
        correct += 1
        per_class_stats[true_label]["correct"] += 1

total   = len(test_paths)
overall_acc = correct / total
print(f"Overall Test Accuracy: {overall_acc:.2%} ({correct}/{total})\n")

# 8. Print per-gesture accuracy
class_accuracies = {}
print("Per-Gesture Accuracy:")
for label, stats in per_class_stats.items():
    total_cls   = stats["total"]
    correct_cls = stats["correct"]
    acc         = correct_cls / total_cls if total_cls else 0
    class_accuracies[label] = acc
    print(f" - {label:15s}: {acc:.2%} ({correct_cls}/{total_cls})")

# 9. Plot per-class accuracy
plt.figure(figsize=(12, 6))
plt.bar(class_accuracies.keys(), class_accuracies.values())
plt.xticks(rotation=90)
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Per-Gesture Accuracy (Test Set, 80/20 split)")
plt.tight_layout()
plt.show()
