# 🚦 Traffic Sign Classifier

An end-to-end AI-powered web application that classifies traffic signs using a Convolutional Neural Network (CNN) with a FastAPI backend and a browser-based frontend.

---

## 🧠 Overview

This project uses a trained CNN model to recognize traffic signs from images.  
Users can upload an image through a web interface, and the model returns:

- Predicted traffic sign
- Confidence score

---

## 🚀 Features

- 🔍 Real-time traffic sign prediction
- 🧠 CNN-based image classification
- ⚡ FastAPI backend for inference
- 🌐 Simple HTML + JavaScript frontend
- 📦 Lightweight model (~3.9 MB)

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **FastAPI**
- **OpenCV**
- **NumPy**
- **HTML, CSS, JavaScript**

---

## 📂 Project Structure

```bash
traffic-sign-classifier/
├── app.py
├── model.h5
├── index.html
├── requirements.txt
```
---

## ⚙️ How It Works

1. User uploads an image via the frontend
2. Image is sent to FastAPI backend
3. Backend:
   - preprocesses image
   - feeds into CNN model
4. Model returns prediction + confidence
5. Result displayed on UI

---

## 🧪 Run Locally

### 1. Clone the repository
git clone https://github.com/ArghyaSharma/traffic-sign-classifier.git
cd traffic-sign-classifier
### 2. Install dependencies
pip install -r requirements.txt
### 3. Run the server
uvicorn app:app –reload
### 4. Open in browser
http://127.0.0.1:8000/docs
OR open `index.html` manually.

---

## 🌍 Deployment

This project is designed to be deployed using platforms like:

- Render (recommended)
- Hugging Face Spaces (alternative)

---

## 📊 Model Details

- Input size: **30x30 RGB images**
- Architecture:
  - Conv2D + MaxPooling layers
  - Dense layers with Dropout
- Output: **43 traffic sign classes**

---

## 📌 Notes

- Dataset used: GTSRB (German Traffic Sign Recognition Benchmark)
- Dataset is **not included** in this repository
- Only trained model is provided for inference

---

## 👨‍💻 Author

**Arghya Sharma**

---

## ⭐ Future Improvements

- Deploy frontend with backend (single service)
- Add drag-and-drop UI
- Show top-3 predictions
- Improve model accuracy further

---

## 📜 License

This project is open-source and free to use.
