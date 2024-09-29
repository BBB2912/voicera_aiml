import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av

# Define sliders to control Canny edge detection thresholds
th1 = st.slider("Threshold1", 0, 1000, 100)
th2 = st.slider("Threshold2", 0, 1000, 200)  # Changed to th2

# Define the callback function that processes each video frame
def callback(frame: av.VideoFrame):
    # Convert the frame to a NumPy array in BGR24 format
    image = frame.to_ndarray(format="bgr24")
    
    # Convert the image to grayscale and apply Canny edge detection
    image = cv2.Canny(image, th1, th2)
    
    # Convert the image back to a 3-channel image (RGB)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Return the processed frame as a new VideoFrame
    return av.VideoFrame.from_ndarray(image, format="bgr24")

# Stream video using the callback function for processing frames
webrtc_streamer(
    key="sample",
    video_frame_callback=callback  # Corrected the parameter name to 'video_frame_callback'
)
