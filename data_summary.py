import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1) CONFIG: point this at your top-level dataset folder
#    which should contain subfolders for each class.
DATA_DIR = "Data"

# 2) COUNT SAMPLES PER CLASS
classes = [d for d in os.listdir(DATA_DIR)
           if os.path.isdir(os.path.join(DATA_DIR, d))]
class_counts = {}
all_files, all_labels = [], []
for cls in classes:
    cls_dir = os.path.join(DATA_DIR, cls)
    imgs = [
        f for f in os.listdir(cls_dir)
        if os.path.isfile(os.path.join(cls_dir, f))
        and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    ]
    class_counts[cls] = len(imgs)
    for img in imgs:
        all_files.append(os.path.join(cls_dir, img))
        all_labels.append(cls)

# 3) TEXT SUMMARY: total per class
print("▶ Total samples per class:")
for cls, cnt in class_counts.items():
    print(f"   • {cls:15s} : {cnt}")

# 4) BAR PLOT: samples per class
plt.figure()
plt.bar(class_counts.keys(), class_counts.values())
plt.xticks(rotation=90)
plt.title("Samples per Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 5) SPLIT INTO TRAIN / VAL / TEST (80/10/10 stratified)
train_files, temp_files, train_lbls, temp_lbls = train_test_split(
    all_files, all_labels, test_size=0.20,
    stratify=all_labels, random_state=42)

val_files, test_files, val_lbls, test_lbls = train_test_split(
    temp_files, temp_lbls, test_size=0.50,
    stratify=temp_lbls, random_state=42)

split_counts = {
    "Train": len(train_files),
    "Validation": len(val_files),
    "Test": len(test_files)
}

# 6) TEXT SUMMARY: splits
print("\n▶ Dataset split counts:")
for split, cnt in split_counts.items():
    print(f"   • {split:10s} : {cnt}")

# 7) BAR PLOT: split distribution
plt.figure()
plt.bar(split_counts.keys(), split_counts.values())
plt.title("Dataset Split Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
