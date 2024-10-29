import cv2
from torchvision.io import video
from ultralytics import YOLO
import pandas as pd

#load yolov8
model = YOLO('yolov8n.pt')
#load video
video_path = './highway_mini.mp4'
cap = cv2.VideoCapture(video_path)
ret = True
#read frame

while ret:
    ret, frame = cap.read()

#detect object

#track object

    result = model.track(frame,persist = True)
#plot result
    frame_result = result[0].plot()


#visualize
    cv2.imshow('frame',frame_result)
    #25millisecond or press q to break
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break