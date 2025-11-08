import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

# Load Model & Labels
model = load_model("sign_model.h5")

actions = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
   # Change if your training had more letters

# Mediapipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
    else:
        return np.zeros(21*3)

# Start Webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False,
                     max_num_hands=1,
                     min_detection_confidence=0.5,
                     min_tracking_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Process Frame with Mediapipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Draw Hand Landmarks
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Extract Keypoints
        keypoints = extract_keypoints(results)

        # Reshape for model → (1,1,63)
        input_data = keypoints.reshape(1, 1, 63)

        # Predict
        prediction = model.predict(input_data)[0]
        predicted_letter = actions[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Display Output
        cv2.putText(frame, f"{predicted_letter} ({confidence:.2f})",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.imshow("Sign Language Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
