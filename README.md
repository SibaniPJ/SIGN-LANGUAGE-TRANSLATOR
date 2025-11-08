# SIGN-LANGUAGE-TRANSLATOR
A machine learning-based sign language alphabet recognition system that uses Mediapipe landmarks and a custom-trained neural network for real-time gesture translation.

## Sign Language Translator (A–Z)

This project is a **Real-Time Sign Language Translator** that detects and recognizes **hand signs (A–Z)** using **MediaPipe Hand Tracking** and a **Neural Network model**. The system uses a webcam to capture hand gestures and displays the predicted alphabet in real-time.

---

## ✨ Features
- Detects **26 English alphabets (A–Z)** using hand gestures
- Works **in real-time** using your webcam
- Uses **Mediapipe** for hand landmark extraction
- Custom **Neural Network model** trained on captured gesture data
- Smooth and fast prediction performance

---

## 👩‍💻 Project Details

| Field | Information |
|------|-------------|
| **Project Title** | Sign Language Translator |
| **Student Name** | *Sibani Priyadarshini Jena* |
| **Model Used** | Mediapipe + Neural Network |
| **Dataset** | Collected manually using webcam |
| **Output Type** | Shows predicted letter live on webcam |

---

## 🧠 Model Workflow

1. **Collect Data**  
   Hand landmark coordinates were captured using Mediapipe and stored in a dataset (`data.npy`).

2. **Train Model**  
   A custom Neural Network (Sequential model) was trained on the collected dataset.

3. **Real-Time Prediction**  
   Webcam feeds live hand gesture frames to the model, predicting the corresponding alphabet.

---

## 🗂️ Project Structure
sign_language_translator/
│
├── data.py # Script to collect dataset
├── train.py # Trains the neural network model
├── app.py # Runs the real-time sign detection
├── asl_model.h5 # Saved trained model
├── data.npy # Collected training data (landmarks + labels)
├── function.py # Helper functions (Mediapipe detection, drawing)
└── README.md # Project documentation

## How to run the project
1. Collect your dataset
        python data.py
2. Train the model
        python train.py
3. Run real-time translator
        python app.py

## 🎥 Output (How It Works)
1.Open the webcam window
2.Show a hand gesture corresponding to A–Z
3.The system will display the predicted letter on screen

       




