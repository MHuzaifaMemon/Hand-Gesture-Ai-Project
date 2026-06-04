# train_model_split.py

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ——— CONFIG ———
DATA_DIR      = "Data"             # your single folder of gesture subfolders
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 20
MODEL_SAVE    = "keras_model_custom.h5"
LABELS_SAVE   = "labels.txt"       # saved as: index<space>label (supports multi-word labels)
LEARNING_RATE = 1e-4
VAL_SPLIT     = 0.2                # 20% of data for validation

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        f"DATA_DIR not found: {DATA_DIR}. Expected a folder containing one subfolder per class."
    )

# ——— DATA GENERATORS WITH SPLIT ———
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VAL_SPLIT,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='training'       # <-- tells it to use the 80% training split
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    subset='validation'     # <-- uses the 20% validation split
)

# ——— MODEL DEFINITION ———
base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=output)

# ——— COMPILE ———
model.compile(
    optimizer=Adam(LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ——— TRAIN ———
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)

# ——— SAVE ———
model.save(MODEL_SAVE)
print(f"✅ Saved trained model to {MODEL_SAVE}")

# Save labels in a stable, model-compatible order.
idx_to_label = sorted(((idx, label) for label, idx in train_gen.class_indices.items()), key=lambda x: x[0])
with open(LABELS_SAVE, "w", encoding="utf-8") as f:
    for idx, label in idx_to_label:
        f.write(f"{idx} {label}\n")
print(f"✅ Saved labels to {LABELS_SAVE}")
