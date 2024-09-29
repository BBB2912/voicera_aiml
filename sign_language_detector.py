import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode

# Suppress TensorFlow warnings if not needed
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Load the TensorFlow model without compiling (since we're only predicting)
model = load_model('action.h5', compile=False)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Define sign language actions
keys_list = [
    "I", "You", "We", "Are", "Hello", "Thank You", "Sorry", "What", "Eat",
]
actions = np.array(keys_list)

def mediapipe_detection(image, model):
    """Run Mediapipe detection on the image."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    """Extract keypoints from Mediapipe results."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        self.sequence = []
        self.current_sign = ""
        self.threshold = 0.8
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        # Convert frame to numpy array
        image = frame.to_ndarray(format="bgr24")

        # Make detections
        image, results = mediapipe_detection(image, self.holistic)

        # Draw landmarks
        if results.face_landmarks:
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        

        # Extract keypoints and make predictions
        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]  # Keep the last 30 frames

        if len(self.sequence) == 30:
            res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
            if res[np.argmax(res)] > self.threshold:
                self.current_sign = actions[np.argmax(res)]
            else:
                self.current_sign = "Unknown"

        # Overlay the detected sign on the image
        cv2.putText(image, f'Detected Sign: {self.current_sign}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return image

# Initialize Streamlit app
st.title("Sign Language Detector")
st.write("This app detects sign language using your webcam feed.")

# Configure WebRTC streamer
webrtc_streamer(
    key="sign-language",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=SignLanguageTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
