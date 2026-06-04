import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# --- Camera setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam/video file.")
    exit()

# Try to bump resolution & FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 224
counter = 0
folder = os.getenv("OUTPUT_DIR", os.path.join("Data", "Nice"))  # e.g. Data/Hello

# Make sure output folder exists
os.makedirs(folder, exist_ok=True)

# Sharpening kernel
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image.")
            break

        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            if 'bbox' in hand:
                x, y, w, h = hand['bbox']
                hImg, wImg, _ = img.shape

                # Crop region with offset
                y1, y2 = max(0, y - offset), min(hImg, y + h + offset)
                x1, x2 = max(0, x - offset), min(wImg, x + w + offset)
                imgCrop = img[y1:y2, x1:x2]

                # White background
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                # Resize & center
                aspectRatio = h / w
                if aspectRatio > 1:
                    scale = imgSize / h
                    wCal = math.ceil(scale * w)
                    imgResized = cv2.resize(imgCrop, (wCal, imgSize), interpolation=cv2.INTER_CUBIC)
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wGap + wCal] = imgResized
                else:
                    scale = imgSize / w
                    hCal = math.ceil(scale * h)
                    imgResized = cv2.resize(imgCrop, (imgSize, hCal), interpolation=cv2.INTER_CUBIC)
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hGap + hCal, :] = imgResized

                # Sharpen edges
                imgWhite = cv2.filter2D(imgWhite, -1, sharpen_kernel)

                # Show
                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)

        # Save on 's' press
        if key == ord("s"):
            counter += 1
            filename = os.path.join(folder, f"Image_{int(time.time())}.png")
            cv2.imwrite(filename, imgWhite)
            print(f"Saved {counter}: {filename}")

except KeyboardInterrupt:
    print("Program interrupted. Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()
