import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Define a Video Transformer that applies Canny Edge Detection
class CannyEdgeTransformer(VideoTransformerBase):
    def __init__(self):
        # You can initialize any variables here if needed
        pass

    def transform(self, frame):
        # Convert the frame to a NumPy array in BGR format
        img = frame.to_ndarray(format="bgr24")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny Edge Detection
        edges = cv2.Canny(blurred, 50, 150)

        # Convert single channel back to three channels to display in Streamlit
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return edges_colored

# Streamlit App Layout
st.title("Live Webcam Feed with Canny Edge Detection")
st.write("This app captures your webcam feed and applies a Canny edge detection filter in real-time.")

# Add a description or instructions if needed
st.markdown("""
- Click on the **Start** button to begin streaming.
- The processed video with Canny edges will be displayed.
- Click on the **Stop** button to end streaming.
""")

# Initialize the WebRTC streamer with the CannyEdgeTransformer
webrtc_streamer(
    key="canny-edge",
    video_processor_factory =CannyEdgeTransformer,
    media_stream_constraints={
        "video": True,
        "audio": False  # Disable audio to reduce bandwidth
    }
)
