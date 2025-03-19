# * Import necessary libraries
import cv2  # OpenCV for video processing
import numpy as np  # NumPy for mathematical operations
import os  # OS for file handling
from matplotlib import pyplot as plt  # Matplotlib for visualization
import time  # Time functions
import mediapipe as mp  # MediaPipe for AI-based hand, face, and pose tracking

# * Initialize MediaPipe models
mp_holistic = mp.solutions.holistic  # Full-body tracking (pose, hands, face)
mp_face_mesh = mp.solutions.face_mesh  # Face mesh model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# * Function to process an image with MediaPipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # Convert image to RGB (MediaPipe needs this format)
    image.flags.writeable = False  # Temporarily make image read-only to improve processing speed
    results = model.process(image)  # Apply MediaPipe model to detect landmarks
    image.flags.writeable = True  ## Allow modifications again after processing
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV display
    return image, results  # Return the processed image and detection results

# * Function to draw landmarks on the detected body parts
def draw_styled_landmarks(image, results):
   # * Set a common color style for all landmarks

    # Face landmarks
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
    #                           mp_drawing.DrawingSpec(color=(80,110,10), thickness=1 , circle_radius=1),
    #                           mp_drawing.DrawingSpec(color=(80,256,121), thickness=1 , circle_radius=1)) 
    # # Body pose
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
    #                           mp_drawing.DrawingSpec(color=(80,22,10), thickness=2 , circle_radius=4),
    #                           mp_drawing.DrawingSpec(color=(80,44,121), thickness=2 , circle_radius=2)) 
    # # Left hand
    # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
    #                           mp_drawing.DrawingSpec(color=(121,22,76), thickness=2 , circle_radius=4),
    #                           mp_drawing.DrawingSpec(color=(121,44,250), thickness=2 , circle_radius=2)) 
    # # Right hand
    # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
    #                           mp_drawing.DrawingSpec(color=(245,117,66), thickness=2 , circle_radius=4),
    #                           mp_drawing.DrawingSpec(color=(245,66,230), thickness=2 , circle_radius=2)) 
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



# * Start webcam and process video
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

# * Extract keypoints (pose, face, hands) from results
def extract_keypoints(results):
    """
    Extracts keypoint coordinates from MediaPipe results for pose, face, and hands.

    Returns:
    - A flattened NumPy array of coordinates for each body part.
    - If no detection is found, returns an array of zeros.

    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
