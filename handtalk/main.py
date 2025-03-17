# * Import necessary libraries
import cv2  # OpenCV for computer vision
import numpy as np  # NumPy for numerical operations
import os  # OS for file handling
from matplotlib import pyplot as plt  # Matplotlib for visualization
import time  # Time for delays or performance tracking
import mediapipe as mp  # MediaPipe for hand tracking (if needed later)

# * Initialize webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")  
        break

    cv2.imshow('OpenCV Feed', frame)  
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()