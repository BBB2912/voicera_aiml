# import cv2
# import numpy as np
# import mediapipe as mp
# from tensorflow.keras.models import load_model
# import streamlit as st

# # Load the model
# model = load_model('action.h5')

# mp_drawing = mp.solutions.drawing_utils
# mp_holistic = mp.solutions.holistic

# # Define sign language actions
# keys_list = [
#     "I", "You", "We", "Are", "Hello", "Thank You", "Sorry", "What", "Eat",
# ]
# actions = np.array(keys_list)

# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = model.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     return image, results

# def extract_keypoints(results):
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
#     face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
#     lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
#     rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
#     return np.concatenate([pose, face, lh, rh])

# # Initialize Streamlit app
# st.title("Sign Language Detector")
# st.write("This app detects sign language using a webcam feed.")

# # Initialize webcam feed
# cap = cv2.VideoCapture(0)

# # Placeholder for the video feed
# frame_placeholder = st.empty()

# # Placeholder for detected sign
# sign_placeholder = st.empty()

# # Control flags
# run_detection = False
# sequence = []
# current_sign = ""
# threshold = 0.8

# # Start button
# if st.button('Start', key='start_button'):
#     run_detection = True

# # Stop button
# if st.button('Stop', key='stop_button'):
#     run_detection = False

# # Streamlit app structure
# if run_detection:
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         while cap.isOpened() and run_detection:
#             ret, frame = cap.read()
#             if not ret:
#                 st.error("Unable to access the webcam.")
#                 break

#             # Make detections
#             image, results = mediapipe_detection(frame, holistic)

#             # Extract keypoints and make predictions
#             keypoints = extract_keypoints(results)
#             sequence.append(keypoints)
#             sequence = sequence[-30:]

#             if len(sequence) == 30:
#                 res = model.predict(np.expand_dims(sequence, axis=0))[0]

#                 if res[np.argmax(res)] > threshold:
#                     current_sign = actions[np.argmax(res)]  # Update the current sign

#             # Display the output
#             frame_placeholder.image(image, channels="BGR", use_column_width=True)
#             sign_placeholder.empty()  # Clear the previous output
#             sign_placeholder.write('Detected Sign: ' + current_sign)  # Overwrite the displayed sign

# # Release resources after the loop
# cap.release()
# cv2.destroyAllWindows()
