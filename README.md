# Hand Gesture Recognition (CV + Keras)

This project trains and runs a hand-gesture classifier using image data in `Data/` (one folder per gesture). It includes:

- **Data capture** from webcam with hand-cropping (`DataCollection.py`)
- **Training** a MobileNetV2-based classifier (`train_model_split.py`)
- **Evaluation** scripts (accuracy, confusion matrix, k-fold)
- **Real-time prediction** from webcam (`test.py`)

## Dataset structure

Your dataset folder must look like this:

```
AI_Project/
  Data/
    Hello/
    Thank you/
    Good Luck/
    ...
```

Each gesture folder contains images for that class.

## Setup (Windows)

1) Create + activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

## 1) Collect data (optional)

This script opens the webcam, detects a hand, crops it, and shows a 224×224 white-background image.

- Press `s` to save an image.
- By default it saves into `Data\Nice`.

Run:

```powershell
python DataCollection.py
```

To save to a different class folder without editing code:

```powershell
$env:OUTPUT_DIR = "Data\Hello"
python DataCollection.py
```

## 2) Train a model

Trains a MobileNetV2 (transfer learning) classifier using an 80/20 train/validation split.

Run:

```powershell
python train_model_split.py
```

Outputs:

- `keras_model_custom.h5` (trained model)
- `labels.txt` (class index → label mapping)

## 3) Run real-time webcam prediction

By default, `test.py` loads:

- `keras_model_custom2.h5` (if present)
- `labels.txt` (if present) or it infers labels from `Data/`

Run:

```powershell
python test.py
```

If you want to use the model you just trained:

```powershell
$env:MODEL_FILE = "keras_model_custom.h5"
$env:LABELS_FILE = "labels.txt"
python test.py
```

Press `q` to quit.

## 4) Evaluation scripts

These scripts load a saved model and run evaluation on the images in `Data/`.

```powershell
python data_summary.py
python accuracy_analysis.py
python confusion_matrix.py
python k-fold-analysis.py
```

Environment overrides (optional):

```powershell
$env:DATA_DIR = "Data"
$env:MODEL_FILE = "keras_model_custom2.h5"
$env:LABELS_FILE = "labels.txt"
python confusion_matrix.py
```

## Notes / Troubleshooting

- If `labels.txt` is missing, the scripts try to infer labels from subfolder names in `Data/` (sorted). For best results, keep `labels.txt` with the model you trained.
- If you push this to GitHub, do **not** commit `venv/` (it is machine-specific).
- `.h5` model files can be large; consider Git LFS if GitHub rejects the push.

## Project files

- `DataCollection.py` — collect cropped hand images into `Data/<ClassName>`
- `train_model_split.py` — train model and export `labels.txt`
- `test.py` — real-time webcam prediction
- `accuracy_analysis.py` — overall + per-class accuracy plot
- `confusion_matrix.py` — confusion matrix + classification report
- `k-fold-analysis.py` — stratified k-fold evaluation for a saved model
- `data_summary.py` — dataset counts and split distribution plots
