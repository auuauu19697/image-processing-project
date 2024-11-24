import cv2
import streamlit as st
from ultralytics import YOLO
import time

# Load the YOLOv8 model
model = YOLO('yolo8n.pt')

def video_process_page():

    # Streamlit app title
    st.title("YOLOv8 Object Detection and Tracking")

    # Video file uploader
    video_file = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'mkv'])

    if video_file is not None:
        # Initialize counts and display placeholders
        car_count = 0
        accident_count = 0
        crossed_car_count = 0
        car_count_text = st.empty()
        accident_count_text = st.empty()
        crossed_car_count_text = st.empty()
        car_count_text.markdown(f"**Car detected:** {car_count}")
        accident_count_text.markdown(f"**Accident detected:** {accident_count}")
        crossed_car_count_text.markdown(f"**Cars crossed the line:** {crossed_car_count}")

        # Define a vertical line position for detecting crossings
        line_position = 240

        # Save uploaded video to a temporary file
        temp_video_path = 'temp_video.mp4'
        with open(temp_video_path, 'wb') as f:
            f.write(video_file.read())

        # Load video and initialize placeholders
        cap = cv2.VideoCapture(temp_video_path)
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
                detections = result[0].boxes
                labels = detections.cls
                confidences = detections.conf
                bboxes = detections.xyxy
                ids = detections.id

                car_count = 0  # Reset car count for this frame

                for i in range(len(labels)):
                    if labels[i] == 1 and confidences[i] > 0.35:  # Only for car detections
                        car_count += 1
                        x1, y1, x2, y2 = bboxes[i].cpu().numpy()
                        car_id = int(ids[i].item())

                        if car_id not in crossed_car_ids:
                            if car_id not in tracked_cars:
                                tracked_cars[car_id] = y1
                            else:
                                previous_y = tracked_cars[car_id]
                                if (previous_y < line_position <= y1) or (previous_y > line_position >= y1):
                                    crossed_car_count += 1
                                    crossed_car_ids.add(car_id)
                            tracked_cars[car_id] = y1

                # Clean up tracked cars
                tracked_cars = {car_id: pos for car_id, pos in tracked_cars.items() if car_id in ids.cpu().numpy()}

                # Convert frame from BGR to RGB
                frame_result = cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB)

                # Update Streamlit display
                stframe.image(frame_result, channels='RGB', use_container_width=True)
                car_count_text.markdown(f"**Car detected:** {car_count}")
                accident_count_text.markdown(f"**Accident detected:** {accident_count}")
                crossed_car_count_text.markdown(f"**Cars crossed the line:** {crossed_car_count}")

                time.sleep(0.015)  # Control frame rate

            frame_counter += 1

        cap.release()  # Release video capture object
