import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
import time

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Streamlit app title
st.title("YOLOv8 Object Detection and Tracking")

# Upload video file
video_file = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'mkv'])

if video_file is not None:
    # Create a temporary file to store the uploaded video
    temp_video_path = 'temp_video.mp4'
    
    with open(temp_video_path, 'wb') as f:
        f.write(video_file.read())

    # Load video
    cap = cv2.VideoCapture(temp_video_path)
    stframe = st.empty()  # Placeholder for displaying frames

    frame_counter = 0
    frame_skip = 2  # Adjust this value to skip frames (e.g., 2 means process every 2nd frame)

    ret = True
    while ret:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Only process every nth frame
        if frame_counter % frame_skip == 0:
            # Optionally resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))  # Resize to 640x480 for faster processing

            # Detect and track objects
            result = model.track(frame, persist=True)

            # Plot result
            frame_result = result[0].plot()

            # Convert BGR (OpenCV format) to RGB (Streamlit format)
            frame_result = cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB)

            # Display frame in Streamlit
            stframe.image(frame_result, channels='RGB', use_container_width=True)  # Updated here
            
            # Sleep to control the frame rate (adjust for desired smoothness)
            time.sleep(0.015)  # Adjust the sleep time as needed

        frame_counter += 1

    cap.release()  # Release video capture object
