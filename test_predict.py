import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model
model = tf.keras.models.load_model("breed_model.h5")

# Load class mapping
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping (index -> class name)
class_names = list(class_indices.keys())

# Load image
img = Image.open("jaffar.jpg").convert("RGB").resize((224, 224))
img = np.array(img)

# IMPORTANT: MobileNetV2 preprocessing
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
img = preprocess_input(img)

img = np.expand_dims(img, axis=0)

# Predict
pred = model.predict(img)
idx = np.argmax(pred)
confidence = np.max(pred)

print("Prediction:", class_names[idx])
print("Confidence:", round(float(confidence), 2))
