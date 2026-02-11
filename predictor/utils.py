import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ----------------------
# Paths
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "breed_model.h5")
CLASS_JSON_PATH = os.path.join(BASE_DIR, "model", "class_indices.json")

# ----------------------
# Load Model
# ----------------------
model = tf.keras.models.load_model(MODEL_PATH)

# ----------------------
# Load Class Mapping
# ----------------------
with open(CLASS_JSON_PATH, "r") as f:
    class_indices = json.load(f)

CLASS_NAMES = list(class_indices.keys())

# ----------------------
# Breed Information (ALL 13 BREEDS)
# ----------------------
BREED_INFO = {

    # 🐄 CATTLE BREEDS
    "Gir": {
        "origin": "Gujarat, India",
        "milk": "1200–1800 liters per lactation",
        "climate": "Hot and dry regions",
        "features": "Long curved horns, red coat, disease resistant",
        "use": "Dairy and breeding"
    },

    "Sahiwal": {
        "origin": "Punjab region",
        "milk": "2000–3000 liters per lactation",
        "climate": "Hot and humid climates",
        "features": "Reddish brown coat, docile nature",
        "use": "High quality milk production"
    },

    "Red_Sindhi": {
        "origin": "Sindh region",
        "milk": "1500–2500 liters per lactation",
        "climate": "Dry and semi-arid regions",
        "features": "Deep red color, strong immunity",
        "use": "Milk production"
    },

    "Hallikar": {
        "origin": "Karnataka, India",
        "milk": "Low (mainly draft breed)",
        "climate": "Hot and dry climates",
        "features": "Strong body, long horns",
        "use": "Draft and farming work"
    },

    "Hariana": {
        "origin": "Haryana, India",
        "milk": "1000–1500 liters per lactation",
        "climate": "Hot plains",
        "features": "White or light grey body",
        "use": "Dual purpose (milk + draft)"
    },

    "Kankrej": {
        "origin": "Gujarat and Rajasthan",
        "milk": "1400–1800 liters per lactation",
        "climate": "Hot and dry climates",
        "features": "Lyre-shaped horns, strong body",
        "use": "Milk and draft"
    },

    "Deoni": {
        "origin": "Maharashtra, India",
        "milk": "1200–1800 liters per lactation",
        "climate": "Tropical climates",
        "features": "Black and white patches",
        "use": "Dual purpose"
    },

    "Tharparkar": {
        "origin": "Rajasthan desert region",
        "milk": "1800–2600 liters per lactation",
        "climate": "Hot and dry desert climates",
        "features": "White coat, heat tolerant",
        "use": "Milk production"
    },

    # 🐃 BUFFALO BREEDS
    "murrah": {
        "origin": "Haryana & Punjab",
        "milk": "2500–3000 liters per lactation",
        "climate": "Moderate to hot climates",
        "features": "Jet black body, tightly curled horns",
        "use": "High fat milk production"
    },

    "surti": {
        "origin": "Gujarat, India",
        "milk": "1500–2500 liters per lactation",
        "climate": "Hot climates",
        "features": "Sickle shaped horns",
        "use": "Milk production"
    },

    "pandharpuri": {
        "origin": "Maharashtra, India",
        "milk": "1800–2200 liters per lactation",
        "climate": "Hot semi-arid climates",
        "features": "Very long horns",
        "use": "Milk production"
    },

    "bhadwari": {
        "origin": "Uttar Pradesh & Madhya Pradesh",
        "milk": "1500–2000 liters per lactation",
        "climate": "Hot climates",
        "features": "Copper colored coat",
        "use": "High fat milk"
    },

    "Jaffarabadi": {
        "origin": "Gujarat, India",
        "milk": "2000–2500 liters per lactation",
        "climate": "Hot coastal regions",
        "features": "Massive body, heavy drooping horns",
        "use": "Milk and draft"
    }
}

# ----------------------
# Image Preprocessing
# ----------------------
def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB").resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# ----------------------
# Prediction Function
# ----------------------
def predict_breed(image_file):
    img = preprocess_image(image_file)
    pred = model.predict(img)[0]

    index = np.argmax(pred)
    breed = CLASS_NAMES[index]
    confidence = round(float(pred[index]) * 100, 2)

    info = BREED_INFO.get(breed, {})

    return {
        "breed": breed,
        "confidence": confidence,
        "info": info
    }
