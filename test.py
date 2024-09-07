import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import os
from datetime import datetime

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

# Open the video
cap = cv2.VideoCapture('wrongway.mp4')

# Load the COCO class names
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Define areas
area1 = [(593,227), (602,279), (785,274), (774,220)]
area2 = [(747,92), (785,208), (823,202), (773,95)]

# Create a folder to save wrong-way car images
save_dir = "wrong_way_cars"
os.makedirs(save_dir, exist_ok=True)

# Car status tracking
car_status = {}
wrong_way_cars = set()  # Set to keep track of wrong-way cars

while True:    
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for consistency
    frame = cv2.resize(frame, (1020, 500))

    # Use YOLO with built-in tracking
    results = model.track(source=frame, persist=True)

    # Counter for wrong-way cars
    wrong_way_count = len(wrong_way_cars)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        classes = result.boxes.cpu().numpy()  # Class IDs
        ids = result.boxes.id.cpu().numpy()  # Object IDs (tracked)
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            obj_class = int(classes[i])
            obj_id = int(ids[i])

            # Check if it's a car (class 'car' is ID 2 in COCO)
            if class_list[obj_class] == 'car':
                cx = (x1 + x2) // 2  # Center x of bounding box
                cy = y2  # Bottom y-coordinate of bounding box

                # Check if car is in area1 or area2
                in_area1 = cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False) >= 0
                in_area2 = cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False) >= 0

                # Initialize car state if not present
                if obj_id not in car_status:
                    car_status[obj_id] = {'in_area1': False, 'in_area2': False, 'wrong_way': False, 'saves': False}

                # Update car's status based on its current position
                if in_area1:
                    car_status[obj_id]['in_area1'] = True
                if in_area2:
                    car_status[obj_id]['in_area2'] = True

                # Check if the car went from area1 to area2 (wrong way)
                if car_status[obj_id]['in_area1'] and in_area2 and not car_status[obj_id]['wrong_way']:
                    car_status[obj_id]['wrong_way'] = True

                    # Save the wrong-way car image only once
                    if not car_status[obj_id]['saved']:
                        # Use current time as the filename
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        car_image_path = os.path.join(save_dir, f'car_{obj_id}_{timestamp}.png')

                        # Crop the car from the frame and save the image
                        car_image = frame[y1:y2, x1:x2]
                        cv2.imwrite(car_image_path, car_image)

                        # Mark the car as saved
                        car_status[obj_id]['saved'] = True

                        # Add to wrong-way cars set
                        wrong_way_cars.add(obj_id)

                # Draw bounding box and wrong way text if applicable
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cvzone.putTextRect(frame, f'ID: {obj_id}', (x1, y1), 1, 1)

                if car_status[obj_id]['wrong_way']:
                    cvzone.putTextRect(frame, f'Wrong Way {obj_id}', (x1, y1 - 20), 1, 1, colorR=(0, 0, 255))

        # Visualize areas
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 255, 255), 2)
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 255, 255), 2)

    # Display the number of wrong-way cars on the frame
    cvzone.putTextRect(frame, f'Wrong Way Cars: {wrong_way_count}', (10, 30), 1, 2, colorR=(0, 255, 0))

    # Write the processed frame to the output video

    # Display the frame
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()