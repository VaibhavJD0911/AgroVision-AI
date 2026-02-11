import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json

# ----------------------
# Configuration
# ----------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 8

# ----------------------
# Data Generators (MobileNetV2 preprocessing)
# ----------------------
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=35,
    zoom_range=0.25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode="nearest"
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_gen.flow_from_directory(
    "dataset/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    "dataset/validation",   # make sure this folder exists
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_data.num_classes

# ----------------------
# Save class indices (VERY IMPORTANT)
# ----------------------
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f, indent=4)

# ----------------------
# Base Model (MobileNetV2)
# ----------------------
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # Stage 1: freeze CNN

# ----------------------
# Custom Classifier Head
# ----------------------
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
])

# ----------------------
# Callbacks
# ----------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-6
)

# ----------------------
# Stage 1 Training
# ----------------------
print("\n🚀 Stage 1: Training classifier layers...\n")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_stage1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_STAGE1,
    callbacks=[early_stop, reduce_lr]
)

# ----------------------
# Stage 2 Fine Tuning
# ----------------------
print("\n🔥 Stage 2: Fine-tuning top CNN layers...\n")

# Unfreeze only top layers (safer)
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_stage2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_STAGE2,
    callbacks=[early_stop, reduce_lr]
)

# ----------------------
# Save Model
# ----------------------
model.save("breed_model.h5")
print("\n✅ Model training complete. Saved as breed_model.h5")
