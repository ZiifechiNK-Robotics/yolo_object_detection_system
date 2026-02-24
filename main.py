'''
YOLO Object Detection and Dataset Generation System
Written by: Ziifechi
Date: February 2026

Project Description:
Using YOLOv8 and OpenCV, I've created a real-time object detection system
It implements custom logic to classify known and unknown objects
Unknown objects are stabilized across multiple frames and saved for future model training

'''

# TINKLE'S BRAIN #

import cv2
import time
from ultralytics import YOLO
import os
from datetime import datetime

# AI model trains on train/ and checks performance on val/ (like training and validation)
# in different angles and lightings, the image (train captures) val still recognizes the image, images that have been captured in the past don't seem so new

# we want to make a training datatset, teach the frames to truly distinguish know and unknown after a 3-5 second count and save to memory/folder


# load YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0) # camera on

SAVE_COOLDOWN = 5 # secs
UNKNOWN_TIME_THRESHOLD = 10 #secs (how long before stable)

last_save_time = 0
unknown_counter = 0
stable_unknown = False
previous_stable_state = False

# folder storing
PLANT_CANDIDATE_DIR = "dataset_brain/plant_candidates"
os.makedirs(PLANT_CANDIDATE_DIR, exist_ok = True)

HUMANS = ["person"]

ENVIRONMENT = [
    "chair",
    "couch",
    "dining table",
    "bottle"
]

PLANT_LIKE_OBJECTS = [
    "potted plant",

    "vase",
    "banana",
    "apple",
    "broccoli",
    "carrot",
    "sweet potato",
    "irish potato",
    "grapes",
    "avacado",
    "beans",
    "rice",
    "pepper",
    "tomatoe",
    "pumpkin seeds",
    "lettuce",
    "onion"
    "bell pepper"
    
    ]

while True:
    ret, frame = cap.read()
    print("ret:",ret)
    print("frame type:",type(frame))
    print("frame shape:", None if frame is None else frame.shape)
    raw_frame = frame.copy() # created because reusing frame could be causing errors with my images
    if not ret:
        break

    results = model(frame, stream=True)
    current_time = time.time()

    detected_unknown = False #reset per frame
    detected_known = False

    for r in results:
        for box in r.boxes:

            cls_id = int(box.cls[0])
            label = model.names[cls_id] 
            confidence = float(box.conf[0])
            print(f"Detected: {label}, Conf: {confidence}")

            if confidence < 0.25:
                continue

            x1,y1,x2,y2 = map(int,box.xyxy[0])

            # cropping only unknown detected object
            crop = frame[y1:y2, x1:x2]


            ## known ##
            if label in HUMANS:
                category = "human"
                detected_known = True
                color = (255,0,0)

            elif label in ENVIRONMENT:
                category = "human"
                detected_unknown = True
                color = (0,255,255)

            elif label in PLANT_LIKE_OBJECTS:
                category = "plant_candidate"
                detected_unknown = True
                color = (0,165,255)

            else:
                category = "unknown"
                detected_unknown = True
                color = (0,0,255) # default color box

            display_text = f"{label}({category})"

            # Draw rec
            cv2.rectangle(frame, (x1,y1),(x2,y2),color,2)

            cv2.putText(
                frame,
                display_text,
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )
            
    # Increase unknown counter if no known object
    if detected_unknown:
        unknown_counter += 1
    else:
        unknown_counter = 0
        stable_unknown = False

    # Confirm stable unknown
    if unknown_counter >= UNKNOWN_TIME_THRESHOLD:
        stable_unknown = True
    else:
        stable_unknown = False

    # Display
    if stable_unknown and not previous_stable_state:

        #if not saved_flag:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{PLANT_CANDIDATE_DIR}/unknown_{timestamp}.jpg"
        
        if frame is not None:
            success = cv2.imwrite(filename, crop) 

            if success:
                print(f"[INFO] Saved plant candidate: {filename}")
            else:
                print("[ERROR] Failed to save image")


        last_save_time = current_time

    previous_stable_state = stable_unknown

    if stable_unknown:
        status_text = "Stable Unknown"
        status_color = (0,0,255) # Red

    else:
        status_text = "KNOWN"
        status_color = (0,255,0) # Green

    cv2.putText(
        frame,
        status_text,
        (50,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2
    )
    previous_stable_state = stable_unknown

    cv2.imshow("Farm Visual Detector/YOLO Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()