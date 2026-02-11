# 🌾 AgroVision-AI  
### AI-Powered Cattle & Buffalo Breed Identification with Farmer Assistant

AgroVision-AI is a deep learning–based web application that identifies **cattle and buffalo breeds from images** and provides guidance through a **farmer-focused chatbot**. The project integrates **computer vision, machine learning, and web development** to support smarter livestock management.

---

## 🚀 Features

- 🐄 **Cattle & Buffalo Breed Identification** using deep learning  
- 🧠 **CNN-based Image Classification** with TensorFlow  
- 🤖 **Rule-Based Farmer Chatbot** for livestock-related queries  
- 🌐 **Web Application** built using Django  
- 📊 Displays **prediction confidence** and breed information  
- 🎥 Complete **video demo** showing real-time working  

---

## 🛠️ Tech Stack

### Backend & Machine Learning
- Python  
- Django  
- TensorFlow / Keras  
- NumPy  
- Pillow  

### Frontend
- HTML  
- CSS  
- JavaScript  

### Tools & Platforms
- Kaggle (dataset source)  
- GitHub (version control)  

---

## 📊 Dataset

This project uses **publicly available datasets from Kaggle** for cattle and buffalo breed image classification.

Due to large size constraints, the datasets are **not included in this repository**.

🔗 **Dataset Sources (Kaggle):**
- Cattle Breed Dataset: https://www.kaggle.com/datasets/sujayroy723/indian-cattle-breeds  
- Buffalo Breed Dataset: https://www.kaggle.com/datasets/atharvadarpude/indian-buffalo-dataset  

After downloading, organize the dataset in the following structure:


dataset/
├── train/
  ├── cattle/
  ├── buffalo/
├── test/
  ├── cattle/
  ├── buffalo/


Place the `dataset/` folder in the project root before training or testing the model.

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/VaibhavJD0911/AgroVision-AI.git
cd AgroVision-AI

2️⃣ Create and activate virtual environment
python -m venv cattle-env

# Windows
cattle-env\Scripts\activate

# macOS / Linux
source cattle-env/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Django server
python manage.py runserver


Open your browser and visit:

http://127.0.0.1:8000/

🧪 Model Training & Testing

Train the model:

python train_model.py


Test predictions:

python test_predict.py

🎥 Video Demo

A full working demo of AgroVision-AI is available here:

▶️ YouTube Demo:
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
