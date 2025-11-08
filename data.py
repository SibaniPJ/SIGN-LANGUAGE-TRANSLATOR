import os
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Path and gesture categories
DATA_PATH = "Image"
actions = os.listdir(DATA_PATH)  # Example: ['A','B','C']
print("Detected Classes:", actions)

data = []  # to store landmark sequences
labels = []  # to store class label index

for idx, action in enumerate(actions):
    folder_path = os.path.join(DATA_PATH, action)
    
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping: {img_path} (not found or unreadable)")
            continue

        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Ensure one hand is detected
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                    landmarks.append(lm.z)

            data.append(landmarks)
            labels.append(idx)

data = np.array(data)
labels = np.array(labels)

print("Data Shape:", data.shape)
print("Labels Shape:", labels.shape)

# Save dataset
np.save("X.npy", data)
np.save("y.npy", labels)

print("✅ Dataset Successfully Created and Saved!")
