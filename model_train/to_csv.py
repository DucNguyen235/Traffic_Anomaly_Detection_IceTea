from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import hdbscan
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import os 
def parse_args():
    parser = argparse.ArgumentParser(description="Image generator from CSV")
    parser.add_argument("video_id", default= "run_all_file", type=str, help="Video ID default will run all video in the folder")

    return parser.parse_args()
# video_name = sys.argv[1]

if __name__ == "__main__":
    args = parse_args()
    video_id = args.video_id
    video_names = []
    if video_id == "run_all_file":
       # list to store files
        res = []

        # Iterate directory
        for file_path in os.listdir(dir_path):
            # check if current file_path is a file
            if os.path.isfile(os.path.join(dir_path, file_path)):
                # add filename to list
                videos_id.append(file_path.split(".")[0])
    else:
        video_names.append(video_id)
    
    for i in videos_id:
        # Load the YOLOv8 model    
        model = YOLO('yolov8x.pt')

        # Open the video file
        video_id = i
        video_path = str(video_id)+'.mp4'
        cap = cv2.VideoCapture(video_path)
        print(video_id)
        # Store the track history
        track_history = defaultdict(lambda: [])

        # Dictionary to store object data
        object_data = defaultdict(list)
        data = []
        # Loop through the video frames
        frame_count = 0
        success, frame = cap.read()

        while success:
            # Read a frame from the video
            success, frame = cap.read()
            if not success:
                break
            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, persist=True, device =0, verbose = False)

                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                if results[0].boxes.id is None:
                    continue
                track_ids = results[0].boxes.id.int().cpu().tolist()
                object_classes = results[0].boxes.cls.int().cpu().tolist()
                confi_scores = results[0].boxes.conf.cpu().tolist()

                # Update object_data dictionary with boxes for each object
                for box, track_id, confi_score,object_class in zip(boxes, track_ids,confi_scores,object_classes):
                    new_x, new_y, w, h = box

                    with open(os.path.join(f'csv/{video_id}.csv'), "a+") as file:
                        line = f"{frame_count}, {track_id}, {new_x}, {new_y}, {w}, {h}, {object_class}, {confi_score}\n"
                        file.write(line)
                # Increment frame count
                frame_count += 1
        # Release the video capture object
        cap.release()
        cv2.destroyAllWindows()

