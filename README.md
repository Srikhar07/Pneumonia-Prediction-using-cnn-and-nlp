# Pneumonia Prediction using CNN and NLP (Multi-Modal Medical Assistant)

## Visit: 

## 📌 Project Overview

This project is an end-to-end AI medical assistant that predicts the likelihood of pneumonia using two sources of information:

* Chest X-ray image analysis using a Convolutional Neural Network (CNN)
* Symptom text analysis using NLP (TF-IDF + Machine Learning)

The system combines both predictions to generate a final confidence score and diagnosis, making it more realistic than single-input models.

The application is deployed as a web app where users can:

* Upload a chest X-ray
* Enter symptoms in text form
* Receive an AI-generated prediction with confidence scores

---

## 🧠 Key Idea

Most academic projects use only image data.
This project uses a **multi-modal approach**:

* Image Model → Detects infection patterns in X-rays
* Symptom Model → Interprets written symptoms
* Final Decision → Combines both predictions

```
Final Score = 0.5 × Image Confidence + 0.5 × Symptom Confidence
```

---

## 🚀 Features

* CNN-based pneumonia detection from chest X-rays
* NLP-based symptom classification using TF-IDF
* Combined prediction for better decision support
* Flask web interface for real-time testing
* TFLite model for lightweight deployment
* User input form (name, age, symptoms, image upload)
* Confidence score breakdown:

  * Image model confidence
  * Symptom model confidence
  * Final combined confidence

---

## 🛠️ Tech Stack

### Machine Learning

* TensorFlow / Keras (CNN training)
* TensorFlow Lite (optimized inference)
* Scikit-learn (symptom classification)
* TF-IDF Vectorization (text processing)
* Joblib (model saving/loading)

### Web Development

* Flask
* HTML + CSS
* Gunicorn (production server)

### Tools

* Python
* NumPy
* Pillow

---

## 📂 Project Structure

```
Pneumonia-Prediction-using-cnn-and-nlp/
│
├── app.py
├── requirements.txt
├── runtime.txt
│
├── model/
│   ├── lung_model.tflite
│   ├── symptom_model.pkl
│   └── tfidf.pkl
│
├── static/
│   ├── uploads/
│   ├── result.css
│
├── templates/
│   ├── home.html
│   └── result.html
│
└── notebooks/
    └── cnn_training.ipynb
```

---

## 🔬 How It Works

### 1) Image Prediction

* Input: Chest X-ray
* Resized to 64×64
* Passed into TFLite CNN model
* Outputs probability of pneumonia

### 2) Symptom Prediction

* Input: User symptom text
* TF-IDF vectorization
* Classified using trained ML model
* Outputs probability of infection

### 3) Final Decision

Both probabilities are combined:

```
Final = (Image Score + Symptom Score) / 2
```

If final ≥ 0.5 → Pneumonia detected
Else → Normal

---

## 🖥️ Running Locally

### 1) Clone repository

```
git clone https://github.com/Srikhar07/Pneumonia-Prediction-using-cnn-and-nlp.git
cd Pneumonia-Prediction-using-cnn-and-nlp
```

### 2) Install dependencies

```
pip install -r requirements.txt
```

### 3) Run the app

```
python app.py
```

Open in browser:

```
http://127.0.0.1:8080
```

---

## 📊 Model Details

### CNN Model

* Trained on chest X-ray dataset
* Binary classification:

  * NORMAL
  * PNEUMONIA
* Converted to TensorFlow Lite for faster inference

### Symptom Model

* TF-IDF feature extraction
* Trained ML classifier using symptom descriptions

---

## 🌐 Deployment

This project can be deployed on:

* Deployed on Render
* Railway
* Any Flask-supported cloud platform

Production server:

```
gunicorn app:app
```

---

## ⚠️ Disclaimer

This project is for educational and research purposes only.
It is not a medical diagnostic tool and should not replace professional medical advice.

---





Give it a star on GitHub!
