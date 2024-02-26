import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
from matplotlib.collections import LineCollection
from ultralytics import YOLO
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate trajectory image from video/stream")
    parser.add_argument("video_id", default= "run_all_videos", type=str, help="Video ID/Stream address (default will run all videos in the folder)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    video_id = args.video_id
    #LIVE VIDEO
    videos_id = []
    model = YOLO("yolo8x.pt")
    if video_id == "run_all_video":
        # directory/folder path
        dir_path = r"..\data\video"

        # list to store files
        res = []

        # Iterate directory
        for file_path in os.listdir(dir_path):
            # check if current file_ path is a file
            if os.path.isfile(os.path.join(dir_path, file_path)):
                # add filename to list
                videos_id.append(file_path)
    else:
        videos_id.append(video_id)
    for video_id in videos_id:
        cap = cv2.VideoCapture(video_id)
        # Traverse all videos
        ret, frame = cap.read()
        dirname=os.path.dirname
        save_dir = os.path.join(dirname(dirname(__file__)), os.path.join('data\train\img'))
        def categorize_angle(angle):
            if 0 <= angle < math.pi/4:
                return 1
            elif math.pi/4 <= angle < math.pi/2:
                return 2
            elif math.pi/2 <= angle < 3*math.pi/4:
                return 3
            elif 3*math.pi/4 <= angle < math.pi:
                return 4
            elif -math.pi <= angle < -3*math.pi/4:
                return 5
            elif -3*math.pi/4 <= angle < -math.pi/2:
                return 6
            elif -math.pi/2 <= angle < -math.pi/4:
                return 7
            else:  # -math.pi/4 <= angle < 0
                return 8


        for i in range(1,9):
            os.makedirs(os.path.join(save_dir,f"{i}"), exist_ok=True)
        #----------------------------------------------------------------------------
        # Consume from YOLO live
        #----------------------------------------------------------------------------
        data = []  # To store the new_x, new_y, and angle_degrees

        track_history = {}  # To store the track history for each track_id

        # Assuming boxes is a list of tuples containing (new_x, new_y, w, h)
        cmap = plt.get_cmap('hsv', lut=256)  # Increase lut for smoother color transitions
        # Remove the axis
        while ret:
            ret, frame = cap.read()
            results = model.track(frame,show = True, device = 0,verbose= False, persist= True, classes = [2,3,7])
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            if results[0].boxes.id is None:
                continue
            track_ids = results[0].boxes.id.int().cpu().tolist()
            object_classes = results[0].boxes.cls.int().cpu().tolist()
            confi_scores = results[0].boxes.conf.cpu().tolist()

            for box, track_id,object_class in zip(boxes, track_ids,object_classes):
                if object_class not in [2,3,5,7]: continue
                if track_id in track_history:
                    old_x, old_y = track_history[track_id][-1][0:2]  # Get the last (x, y) from track history
                    new_x, new_y, w, h = box
                    angle_degrees = (math.atan2(new_y - old_y, new_x - old_x))

                    data.append([new_x, new_y, angle_degrees,object_class])
                    track_history.setdefault(track_id, []).append((box[0], box[1],angle_degrees))
                else:
                    track_history.setdefault(track_id, []).append((box[0], box[1]))  # Store the current (x, y)



            #----------------------------------------------------------------------------
            # Get each trajectory
            #----------------------------------------------------------------------------
            for trajectoryId,traj in track_history.items():
                if len(traj)< 100:
                    continue
                # Convert the trajectory to a Pandas dataframe
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

                # Draw the line segments on a plane with different colors based on the arctan angle of each segment
                norm = plt.Normalize(-np.pi, np.pi)
                plt.gca().invert_yaxis()
                lc = LineCollection(segments,  linewidth=8, cmap=cmap, norm=norm , array = colors)
                plt.gca().add_collection(lc)
                 # Assuming traj is your trajectory data
                first_point = traj.iloc[0]
                last_point = traj.iloc[-1]

                dx = last_point[0] - first_point[0]
                dy = last_point[1] - first_point[1]

                angle_degrees = math.atan2(dy, dx)

                plt.axis('off')
                plt.savefig(os.path.join(save_dir,f'{trajectoryId}.png'), bbox_inches='tight')

                # Close the plot
                plt.close()
                plt.clf()