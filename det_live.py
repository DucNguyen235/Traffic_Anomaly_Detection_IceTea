import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from matplotlib.collections import LineCollection
from ultralytics import YOLO
from sklearn.svm import OneClassSVM
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import joblib
import argparse


#----------------------------------------------------------------------------
#Tuning
#----------------------------------------------------------------------------
#Trajectory model
trajectory_max = 500
trajectory_min = 150
track_timer_reset = 50
anomaly_threshold = 30
#background model
threshold_iou = 5
history = 0 
varThreshold = 150
alert_time = 300
#This both check if the object disappeared for a set amount of frame then detele them
#can be set individually 
max_frame_gap = track_timer_reset 

#----------------------------------------------------------------------------
#Functions
#----------------------------------------------------------------------------
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = min(x1 + w1, x2 + w2) - x_intersection
    h_intersection = min(y1 + h1, y2 + h2) - y_intersection

    if w_intersection <= 0 or h_intersection <= 0:
        return 0.0

    intersection_area = w_intersection * h_intersection
    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou
def parse_args():
    parser = argparse.ArgumentParser(description="Object Tracking and Anomaly Detection")
    parser.add_argument("video_id", type=str, help="Video ID")
    parser.add_argument("model_path", type=str, help="Path to the pretrained model")
    parser.add_argument("bg_check", type=str, nargs='?', default="false", help="Detection for stationary object(true/false)")
    parser.add_argument("bg_show", type=str, nargs='?', default="false", help="Show the background(true/false)")
    parser.add_argument("trajectory_show", type=str, nargs='?', default="false", help="Show the tracking process (true/false)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    video_id = args.video_id
    bg_check = args.bg_check
    model_path = args.model_path
    bg_show = args.bg_show
    trajectory_show = args.trajectory_show
    #SHOW
    if bg_show == "true":
        bg_show = True
    else: bg_show = False
    if trajectory_show == "true":
        trajectory_show = True
    else: trajectory_show = False

    #----------------------------------------------------------------------------
    #LOAD MODEL AND MATERIAL
    #----------------------------------------------------------------------------

    #Yolo Model
    model = YOLO("model/yolov8x.pt")

    #pretrain model
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    one_class_svm = joblib.load(model_path)

    #background from MOG
    img1 = cv2.imread('data/bg/'+str(video_id)+'.jpg')
    gray_img1= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=False)
    bs = cv2.createBackgroundSubtractorMOG2(100, 16, False)
    bg_subtractor.apply(gray_img1)
    #mask
    #----------------------------------------------------------------------------
    #Initialize variables
    #----------------------------------------------------------------------------

    #For trajectory prediction
    data = []  # To store the new_x, new_y, and angle_degrees
    track_history = {}  # To store the track history for each track_id
    out_of_frame = {}
    cmap = plt.get_cmap('hsv', lut=256)  # Increase lut for smoother color transitions
    norm = plt.Normalize(-np.pi, np.pi)
    track_visibility = {}
    anomaly_count_dict = {}
    consecutive_anomaly_count_dict = {}

    #For background
    object_counts = {}
    object_last_frame = {}
    frame_number = 0

    #----------------------------------------------------------------------------
    #READ VIDEO
    #----------------------------------------------------------------------------
    cap = cv2.VideoCapture("data/video/"+str(video_id) + '.mp4')
    ret, frame = cap.read()
    try:
        mask = cv2.imread("data/mask/" + str(video_id) + '.jpg', 0)
        mask[mask > 0] = 1
        has_mask = True
    except:
        has_mask= False
    print("video:",video_id)
    print("has_mask:", has_mask)    
    print("background detection:", bg_check)
    print("background imageshow:", bg_show)
    print("detection show:", trajectory_show)
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        #----------------------------------------------------------------------------
        #==========================TRAJECTORY PREDICTION=============================
        #----------------------------------------------------------------------------

        #----------------------------------------------------------------------------
        #READ AND RECORD BOUNDING BOX FROM VIDEO
        #----------------------------------------------------------------------------
        results = model.track(frame,show = trajectory_show, device = 0,verbose= False, persist= True)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        current_ids = None
        remove_ids = []
        if results[0].boxes.id is None:
            continue
        track_ids = results[0].boxes.id.int().cpu().tolist()
        object_classes = results[0].boxes.cls.int().cpu().tolist()
        confi_scores = results[0].boxes.conf.cpu().tolist()
        current_ids = track_ids
        for box, track_id,object_class in zip(boxes, track_ids,object_classes):
            x, y, w, h = box
            if has_mask and mask[y,x] == 0:
                continue
            if object_class == 0 or object_class == 3:
                continue
            if track_id in track_history:
                old_x, old_y = track_history[track_id][-1][0:2]  # Get the last (x, y) from track history
                new_x, new_y, w, h = box
                angle_degrees = (math.atan2(new_y - old_y, new_x - old_x))
                # angle_degrees = float(frame_count)

                data.append([new_x, new_y, angle_degrees,object_class])
                track_history.setdefault(track_id, []).append((box[0], box[1],angle_degrees))
            else:
                track_history.setdefault(track_id, []).append((box[0], box[1]))  # Store the current (x, y)

        #----------------------------------------------------------------------------
        # Get each trajectory
        #----------------------------------------------------------------------------
        for trajectoryId,traj in track_history.items():
            if len(traj)< trajectory_min:
                continue
            if len(traj) >= trajectory_max:
                traj.pop(0)
            traj = pd.DataFrame(traj)
            # Set the x-axis and y-axis limits
            plt.xlim(0, frame.shape[1])
            plt.ylim(0, frame.shape[0])
            plt.axis('off')

            # Create a list of line segments with different colors based on the arctan angle of each segment
            segments = []
            colors = []
            for i,row in traj.iterrows():
                if i < len(traj) - 1:
                    next_row = traj.iloc[i + 1]
                    segment = [(row[0] , row[1]), (next_row[0], next_row[1])]
                    dx = next_row[0] - row[0]
                    dy = next_row[1] - row[1]
                    angle = math.atan2(dy, dx)

                    segments.append(segment)
                    colors.append(angle)


            plt.gca().invert_yaxis()
            lc = LineCollection(segments,  linewidth=8, cmap=cmap, norm=norm , array = colors)
            plt.gca().add_collection(lc)
            plt.savefig(f'trajectory_image/{trajectoryId}.png', bbox_inches='tight')
            plt.clf()
        
        #----------------------------------------------------------------------------
        #Prediction
        #----------------------------------------------------------------------------
        for id,traj in track_history.items():
            try:
                new_image_path = f'trajectory_image/{id}.png'  # Change to the path of your new image
                img = image.load_img(new_image_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                # Extract features from the new image
                new_image_feature = base_model.predict(x)
                # Flatten the feature vector for the new image
                new_image_feature_flat = new_image_feature.reshape(1, -1)
                # Make a prediction using the One-Class SVM
                prediction = one_class_svm.predict(new_image_feature_flat)
                # Interpret the prediction
                if prediction[0] == 1:
                    if id in anomaly_count_dict:
                        del anomaly_count_dict[id]
                        del consecutive_anomaly_count_dict[id]
                else:
                    if id in anomaly_count_dict:
                        anomaly_count_dict[id] += 1
                    else:
                        anomaly_count_dict[id] = 0
                        consecutive_anomaly_count_dict[id] = 0
                # Check if it exceeds the anomaly_threshold
                    if anomaly_count_dict[id] >= anomaly_threshold:
                        print(f"ID: {id} is an anomaly.")
                        if id in consecutive_anomaly_count_dict:
                            consecutive_anomaly_count_dict[id] += 1
                        else:
                            consecutive_anomaly_count_dict[id] = 0

                        # If it's consecutive for 5 times, consider it an anomaly
                        if consecutive_anomaly_count_dict[id] >= 10:
                            del anomaly_count_dict[id]
                            del consecutive_anomaly_count_dict[id]
                            with open("anomaly.csv", "a+") as file:
                                line = f"{id}, anomaly\n"
                                file.write(line)
                    else:
                        if id in consecutive_anomaly_count_dict:
                            consecutive_anomaly_count_dict[id] = 0
            except:
                pass
        #----------------------------------------------------------------------------
        #Track Visibility Check
        #----------------------------------------------------------------------------
        for track_id in current_ids:
            if track_id not in track_visibility:
                track_visibility[track_id] = 0
        for track_id in track_visibility.keys():
            if track_id not in current_ids:
                track_visibility[track_id] += 1
                if track_id in track_history and track_visibility[track_id] >= track_timer_reset:
                    del track_history[track_id]
                    track_visibility[track_id] = 0


        #----------------------------------------------------------------------------
        #==============================BACKGROUND====================================
        #----------------------------------------------------------------------------
        if bg_check == 'true':
            fg_mask1 = bs.apply(frame)
            new_bg = bs.getBackgroundImage()
            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(new_bg, cv2.COLOR_BGR2GRAY)
            # Apply background subtraction on the frame
            fg_mask = bg_subtractor.apply(gray_frame, learningRate=0)
            # Find contours of white regions in the image
            contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Extract the largest contour
            threshold_area = 50
            large_contours = [c for c in contours if cv2.contourArea(c) > threshold_area]
            valid_bounding_boxes = []

            for contour in large_contours:
                x, y, w, h = cv2.boundingRect(contour)
                current_box = (x, y, w, h)
                
                # Check if this box overlaps with any of the valid bounding boxes
                is_same_object = False
                for valid_box in valid_bounding_boxes:
                    iou = calculate_iou(valid_box, current_box)
                    if iou > threshold_iou:
                        is_same_object = True
                        break
                
                if not is_same_object:
                    valid_bounding_boxes.append(current_box)
                    cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), 2)

                    # Use a unique identifier for each object (e.g., based on coordinates)
                    object_id = (x, y, w, h)

                    # Update the last frame in which this object was detected
                    object_last_frame[object_id] = frame_number

                    # Increment the count for this object in the dictionary
                    object_counts[object_id] = object_counts.get(object_id, 0) + 1

            # Check for objects that have disappeared
            disappeared_objects = [object_id for object_id, last_frame in object_last_frame.items() if frame_number - last_frame > max_frame_gap]

            # Remove disappeared objects from the dictionary
            for object_id in disappeared_objects:
                del object_counts[object_id]
                del object_last_frame[object_id]

            #check bounding boxes IOU
            #if not fit delete
            

            #  Draw the large contours on a new image
            output_img = np.zeros_like(fg_mask)


            cv2.drawContours(output_img, large_contours, -1, 255, -1)

            # Apply the mask to the original image
            result = cv2.bitwise_and(fg_mask, output_img)

            # Display the output image
            if bg_show:
                fg_mask_img = cv2.resize(fg_mask, (960, 540))  
                frame_img = cv2.resize(frame, (960, 540))  
                cv2.imshow('fg_mask', fg_mask_img)
                cv2.imshow('frame', frame_img)

            for object_id, count in object_counts.items():
                if count >= alert_time:
                    print("Anomaly detected at "+ str(object_id))
                    object_counts[object_id] = 0
            frame_number += 1

            # Wait for a key press and then close all windows
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            else: 
                pass

    # cap.release()
    cv2.destroyAllWindows()
