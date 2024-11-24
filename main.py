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

    # Track car information
    tracked_cars = {}  # Key: car ID (from YOLO), Value: last known position (Y-coordinate)
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
            ids = detections.id  # Get the unique IDs assigned by the model

            car_count = 0  # Reset car count for this frame

            for i in range(len(labels)):
                label = labels[i]
                confidence = confidences[i]
                
                if label == 1 and confidence > 0.35:  # Check if the label is car and confidence is high
                    car_count += 1
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = bboxes[i].cpu().numpy()
                    car_id = int(ids[i].item())  # Use the model's assigned ID
                    
                    # Check if the car has crossed the line (vertical line_position)
                    if car_id not in crossed_car_ids:  # Only count if the car hasn't crossed before
                        # Check for crossing from below to above the line
                        if car_id not in tracked_cars:
                            tracked_cars[car_id] = y1  # Save initial position
                        else:
                            previous_y = tracked_cars[car_id]
                            if previous_y < line_position and y1 >= line_position or previous_y > line_position and y1 <= line_position:
                                # Car has crossed the line, increase the count
                                crossed_car_count += 1
                                crossed_car_ids.add(car_id)  # Mark this car as crossed

                        # Update the previous Y position of the car for the next frame
                        tracked_cars[car_id] = y1

                    # Optionally, draw bounding boxes and labels on the frame
                    # cv2.rectangle(frame_result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # cv2.putText(frame_result, f'ID: {car_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Remove IDs of cars that are not detected in this frame
            tracked_cars = {car_id: pos for car_id, pos in tracked_cars.items() if car_id in ids.cpu().numpy()}

            # Draw the virtual line on the frame (for visual indication)
            # cv2.line(frame_result, (0, line_position), (640, line_position), (0, 0, 255), 2)  # Red line

            # Convert BGR (OpenCV format) to RGB (Streamlit format)
            frame_result = cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB)

            # Display frame in Streamlit
            stframe.image(frame_result, channels='RGB', use_container_width=True)

            # Update the object counts on the Streamlit app (separate text display)
            car_count_text.markdown(f"**Car detected:** {car_count}")
            accident_count_text.markdown(f"**Accident detected:** {accident_count}")
            crossed_car_count_text.markdown(f"**Cars crossed the line:** {crossed_car_count}")

            # Sleep to control the frame rate (adjust for desired smoothness)
            time.sleep(0.015)  # Adjust the sleep time as needed

        frame_counter += 1

    cap.release()  # Release video capture object
