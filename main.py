import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
import time

# Load model
model = YOLO('yolo8n.pt')

st.title("YOLOv8 Object Detection and Tracking")

# Upload video file
video_file = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'mkv'])

# Create placeholders to show the results (counts)
car_count = 0
accident_count = 0
crossed_car_count = 0  # Track how many cars have crossed the line
car_count_text = st.empty()  # Placeholder for car count
accident_count_text = st.empty()  # Placeholder for accident count
crossed_car_count_text = st.empty()  # Placeholder for cars that crossed the line
car_count_text.markdown(f"**Car detected:** {car_count}")
accident_count_text.markdown(f"**Accident detected:** {accident_count}")
crossed_car_count_text.markdown(f"**Cars crossed the line:** {crossed_car_count}")

# Define a virtual line for detecting crossing (you can adjust this)
line_position = 240  # Vertical line position in the frame (can be adjusted)

if video_file is not None:
    # Create a temporary file to store the uploaded video
    temp_video_path = 'temp_video.mp4'
    
    with open(temp_video_path, 'wb') as f:
        f.write(video_file.read())

    # Load video
    cap = cv2.VideoCapture(temp_video_path)
    stframe = st.empty()  # Placeholder for displaying frames

    frame_counter = 0
    frame_skip = 1  # Adjust this value to skip frames (e.g., 2 means process every 2nd frame)

    # Store car tracking information (for tracking crossing)
    tracked_cars = {}  # Key: car ID, Value: previous Y position (to track direction)
    crossed_car_ids = set()  # Set of car IDs that have crossed the line

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

            # Extract detected objects' information
            detections = result[0].boxes  # Get bounding box information
            labels = detections.cls  # Class indices
            confidences = detections.conf  # Confidence scores
            bboxes = detections.xyxy  # Bounding boxes (x1, y1, x2, y2)

            # Example logic: Count specific objects (e.g., "person" or "car")
            car_count = 0
            accident_count = 0
            for i in range(len(labels)):
                label = labels[i]
                confidence = confidences[i]
                
                if label == 1:  # Car class (COCO class 1 - assuming label 1 is "car")
                    car_count += 1
                    
                    # Track the car's position (check for crossing the line)
                    x1, y1, x2, y2 = bboxes[i].cpu().numpy()

                    # Assign a unique ID to each car (for simplicity, using the index here)
                    car_id = f"car_{i}"

                    # Check if the car is already being tracked (based on its previous position)
                    if car_id not in tracked_cars:
                        tracked_cars[car_id] = y1  # Save initial position

                    # Check if the car has crossed the line (vertical line_position)
                    if car_id not in crossed_car_ids:  # Only count if the car hasn't crossed before
                        # Check for crossing from below to above the line
                        if tracked_cars[car_id] < line_position and y1 >= line_position:
                            # Car has crossed the line, increase the count
                            crossed_car_count += 1
                            crossed_car_ids.add(car_id)  # Mark this car as crossed
                            
                    # Update the previous Y position of the car for the next frame
                    tracked_cars[car_id] = y1

                    # Optionally, draw bounding boxes and labels on the frame
                    cv2.rectangle(frame_result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Draw the virtual line on the frame (for visual indication)
            # cv2.line(frame_result, (0, line_position), (640, line_position), (255, 0, 0), 2)  # Red line

            # Convert BGR (OpenCV format) to RGB (Streamlit format)
            frame_result = cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB)

            # Display frame in Streamlit
            stframe.image(frame_result, channels='RGB', use_container_width=True)  # Updated here

            # Update the object counts on the Streamlit app (separate text display)
            car_count_text.markdown(f"**Car detected:** {car_count}")
            accident_count_text.markdown(f"**Accident detected:** {accident_count}")
            crossed_car_count_text.markdown(f"**Cars crossed the line:** {crossed_car_count}")

            # Sleep to control the frame rate (adjust for desired smoothness)
            time.sleep(0.015)  # Adjust the sleep time as needed

        frame_counter += 1

    cap.release()  # Release video capture object
