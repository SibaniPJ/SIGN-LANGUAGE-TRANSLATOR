import cv2
import os

# Webcam
cap = cv2.VideoCapture(0)

# Absolute path to Image folder
directory = r"C:\Users\VICTUS\Desktop\sign_language_translator\Image"

# Create folders A-Z if they do not exist
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    os.makedirs(os.path.join(directory, letter), exist_ok=True)

print("Press A to Z keys to capture images for that letter")
print("Press Q to quit\n")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 255, 255), 2)
    cv2.putText(frame, "Press A-Z to Save | Q to Quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

    cv2.imshow("Image Collection", frame)
    key = cv2.waitKey(1)

    if key == ord('q') or key == ord('Q'):
        print("Saving and exiting...")
        break

    # Check if a key corresponds to A-Z
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if key == ord(letter.lower()):
            # Count existing images to avoid overwriting
            count = len(os.listdir(os.path.join(directory, letter)))
            save_path = os.path.join(directory, letter, f"{letter}_{count}.jpg")
            
            roi = frame[100:400, 100:400]
            cv2.imwrite(save_path, roi)
            print(f"Saved: {save_path}")

cap.release()
cv2.destroyAllWindows()
