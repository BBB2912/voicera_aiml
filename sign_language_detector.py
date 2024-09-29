import  streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av


th1=st.slider("threshhold1",0,1000,100)
th1=st.slider("threshhold1",0,1000,100)
def callback(frame: av.VideoFrame):
    image=frame.to_ndarray(format="bgr24")
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image=cv2.Canny(image,th1,th2)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return av.VideoFrame.from_ndarray(image,format="bgr24")

webrtc_streamer(
    key="sample",
    vedio_frame_callback=callback
)
