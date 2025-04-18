
# * ==========================================
# * 1. Import Dependencies
# * ==========================================
import cv2  # OpenCV for video processing
import numpy as np  # NumPy for mathematical operations
import os  # OS for file handling
from matplotlib import pyplot as plt  # Matplotlib for visualization
import time  # Time functions
import mediapipe as mp  # MediaPipe for AI-based hand, face, and pose tracking
from train_model import (model, x_test) # Import your trained model from train_model.py
from config import actions

# * Initialize MediaPipe models
mp_holistic = mp.solutions.holistic  # Full-body tracking (pose, hands, face)
mp_face_mesh = mp.solutions.face_mesh  # Face mesh model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# * ==========================================
# * 2. Detect Face, Hand and Pose Landmarks
# * ==========================================
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # Convert image to RGB (MediaPipe needs this format)
    image.flags.writeable = False  # Temporarily make image read-only to improve processing speed
    results = model.process(image)  # Apply MediaPipe model to detect landmarks
    image.flags.writeable = True  ## Allow modifications again after processing
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV display
    return image, results  # Return the processed image and detection results

# * Function to draw landmarks on the detected body parts
def draw_styled_landmarks(image, results):
    # Face landmarks
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=1 , circle_radius=1),
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=1 , circle_radius=1)) 
    # Body pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2 , circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2 , circle_radius=2)) 
    # Left hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2 , circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2 , circle_radius=2)) 
    # Right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2 , circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2 , circle_radius=2)) 


# * ==========================================
# * 3. Extract Keypoints
# * ==========================================
# * Extract keypoints (pose, face, hands) from results
def extract_keypoints(results):
    # * Extracts keypoint coordinates from MediaPipe results for pose, face, and hands.
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,face,lh,rh])




## * ==========================================
# * 8. Make Sign Language Predictions
# * ==========================================

# * Make predictions on the test set
res = model.predict(x_test)  # Make predictions on the test set (X_test should be defined in the previous steps)
predicted_action = actions[np.argmax(res[4])]  # Get the predicted action for the 5th test sample (for example)
print(f"Predicted Action: {predicted_action}")


# * ==========================================
# * 9. Save Model Weights
# * 10. Evaluation using a Confusion Matrix
# * 11. Test in Real Time
# * (Coming soon in separate training and inference scripts)


cap = cv2.VideoCapture(0)  

# * Initialize MediaPipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")  # Handle errors
            break

        # * Process and detect landmarks
        image, results = mediapipe_detection(frame, holistic)
        print(results)  # Print detection output

        draw_styled_landmarks(image, results)  # Draw landmarks

        cv2.imshow('OpenCV Feed', image)  # Show processed video
        
        # * Exit on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# * Release resources
cap.release()
cv2.destroyAllWindows()

pose = []
for res in results.pose_landmarks.landmark:
     test =np.array([res.x, res.y, res.z, res.visibility])
     pose.append(test)

