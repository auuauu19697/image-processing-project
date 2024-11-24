import os
import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import time

# Load the YOLOv8 model
model = YOLO('yolo8n.pt')

def video_process_page():

    # Streamlit app title
    st.title("Car Crash Detection and Tracking")

    # Video file uploader
    video_file = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'mkv'])

    if "temp_video_path" not in st.session_state:
        st.session_state.temp_video_path = None  # Initialize session state for the video path

    if video_file is not None:
        # Save uploaded video to a unique temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            st.session_state.temp_video_path = temp_file.name
            temp_file.write(video_file.read())

        try:
            # Load video and initialize placeholders
            cap = cv2.VideoCapture(st.session_state.temp_video_path)
            stframe = st.empty()
            frame_counter = 0
            frame_skip = 1  # Process every nth frame
            tracked_cars = {}
            crossed_car_ids = set()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_counter % frame_skip == 0:
                    frame = cv2.resize(frame, (640, 480))
                    result = model.track(frame, persist=True)
                    frame_result = result[0].plot()

                    # Convert frame from BGR to RGB
                    frame_result = cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB)

                    # Update Streamlit display
                    stframe.image(frame_result, channels='RGB', use_column_width=True)

                    time.sleep(0.015)  # Control frame rate

                frame_counter += 1

            cap.release()  # Release video capture object

        finally:
            # Delete the temporary file after use
            if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
                os.remove(st.session_state.temp_video_path)
                st.session_state.temp_video_path = None  # Reset session state
