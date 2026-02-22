🌾 AgroVision-AI
AI-Powered Cattle & Buffalo Breed Identification System

AgroVision-AI is an AI-driven web application designed to assist farmers and agricultural stakeholders by identifying cattle and buffalo breeds from images and providing basic livestock guidance through a chatbot.

The system combines deep learning (computer vision) with a Django-based web platform, demonstrating a practical application of AI in the agriculture domain.

📌 Project Overview

AgroVision-AI uses a Convolutional Neural Network (CNN) to analyze uploaded images and predict the breed of cattle or buffalo. Alongside breed identification, the application includes a rule-based farmer assistant chatbot to answer common livestock-related questions.

This project showcases skills in:

Machine Learning & Deep Learning

Computer Vision

Backend Web Development

AI-driven Application Design

🚀 Key Features

🐄 Cattle & Buffalo Breed Identification from images

🧠 CNN-based Image Classification using TensorFlow & Keras

📊 Prediction Confidence Display for better interpretability

🤖 Rule-Based Farmer Chatbot for livestock guidance

🌐 Django Web Application with simple and clean UI

🎥 Complete Video Demo showing real-time functionality

🛠️ Technology Stack
Backend & Machine Learning

Python

Django

TensorFlow / Keras

NumPy

Pillow

Frontend

HTML

CSS

JavaScript

Tools & Platforms

Kaggle (Dataset source)

GitHub (Version control)

📊 Dataset Information

This project uses publicly available datasets from Kaggle for cattle and buffalo breed classification.

⚠️ Note: Due to size limitations, datasets are not included in this repository.

Dataset Sources

Indian Cattle Breeds Dataset
https://www.kaggle.com/datasets/sujayroy723/indian-cattle-breeds

Indian Buffalo Dataset
https://www.kaggle.com/datasets/atharvadarpude/indian-buffalo-dataset

Expected Dataset Structure
dataset/
├── train/
│   ├── cattle/
│   └── buffalo/
├── test/
│   ├── cattle/
│   └── buffalo/

Place the dataset/ folder in the project root directory before training or testing the model.

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/VaibhavJD0911/AgroVision-AI.git
cd AgroVision-AI
2️⃣ Create & Activate Virtual Environment
python -m venv cattle-env

Windows

cattle-env\Scripts\activate

macOS / Linux

source cattle-env/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Django Server
python manage.py runserver

Open your browser and visit:
👉 http://127.0.0.1:8000/

🧪 Model Training & Testing
Train the Model
python train_model.py
Test Breed Prediction
python test_predict.py
🎥 Video Demonstration

A complete working demo of AgroVision-AI is available here:

▶️ YouTube Demo
https://youtu.be/OZuFYd-LAIM

📁 Project Structure
AgroVision-AI/
│
├── Agrovision_AI/        # Django project settings
├── predictor/           # Breed identification module
├── chatbot/             # Farmer chatbot module
├── train_model.py       # CNN training script
├── test_predict.py      # Model testing script
├── requirements.txt
├── README.md
└── .gitignore
📌 Future Enhancements

Improve model accuracy with a larger and more diverse dataset

Replace rule-based chatbot with an NLP-based conversational AI

Deploy the application on cloud platforms (AWS / Azure / GCP)

Add mobile-friendly UI and multilingual support

👤 Author

Vaibhav J D
AI & Full-Stack Development Enthusiast
