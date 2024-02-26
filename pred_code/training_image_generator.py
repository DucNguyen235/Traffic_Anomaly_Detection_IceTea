import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
from matplotlib.collections import LineCollection
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Image generator from CSV")
    parser.add_argument("video_id", default= "run_all_file", type=str, help="Video ID")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    video_id = args.video_id
    videos_id = []
    if video_id == "run_all_file":
        # directory/folder path
        dir_path = r"..\data\train\csv"

        # list to store files
        res = []

        # Iterate directory
        for file_path in os.listdir(dir_path):
            # check if current file_path is a file
            if os.path.isfile(os.path.join(dir_path, file_path)):
                # add filename to list
                videos_id.append(file_path.split(".")[0])
    else:
        videos_id.append(video_id)
    # Load your CSV data
    for video_id in videos_id:
        dirname=os.path.dirname
        path = os.path.join(dirname(dirname(__file__)), os.path.join('data\train\csv', str(video_id)+".csv"))
        df = pd.read_csv(path,header=None)
        save_dir = os.path.join(dirname(dirname(__file__)), os.path.join('data\train\image'))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Creating example Pandas Series
        frame_counts = df[0]
        track_ids = df[1]
        new_x_values = df[2]
        new_y_values = df[3]
        w_values = df[4]
        h_values = df[5]
        object_classes = df[6]
        confi_scores = df[7]
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
        # Fake Consume from YOLO live
        #----------------------------------------------------------------------------
        data = []  # To store the new_x, new_y, and angle_degrees
        track_history = {}  # To store the track history for each track_id

        # Assuming boxes is a list of tuples containing (new_x, new_y, w, h)
        boxes = zip(new_x_values, new_y_values, w_values, h_values)


        for box, track_id, frame_count,object_class in zip(boxes, track_ids, frame_counts,object_classes):
            if object_class not in [2,3,5,7]: continue
            if track_id in track_history:
                old_x, old_y = track_history[track_id][-1][0:2]  # Get the last (x, y) from track history
                new_x, new_y, w, h = box
                angle_degrees = (math.atan2(new_y - old_y, new_x - old_x))
                # angle_degrees = float(frame_count)

                data.append([new_x, new_y, angle_degrees,object_class])
                track_history.setdefault(track_id, []).append((box[0], box[1],angle_degrees))
            else:
                track_history.setdefault(track_id, []).append((box[0], box[1]))  # Store the current (x, y)


        cmap2 = plt.get_cmap('hsv', lut=256)

        #----------------------------------------------------------------------------
        # Get each trajectory
        #----------------------------------------------------------------------------
        for trajectoryId,traj in track_history.items():
            if len(traj)< 100:
                continue
        # for trajectoryId in [4,6,9,13,14,15]:
            # traj = pd.DataFrame(track_history[trajectoryId])
            traj = pd.DataFrame(traj)

            # Set the x-axis and y-axis limits
            plt.xlim(0, 3840)
            plt.ylim(0, 2160)

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
                    # angle = ((dy/ dx) *180 )/ math.pi
                    # if (angle < 0):
                    #     angle = 
                    segments.append(segment)
                    colors.append(angle)

            # Draw the line segments on a plane with different colors based on the arctan angle of each segment
            norm = plt.Normalize(-np.pi, np.pi)
            plt.gca().invert_yaxis()
            lc = LineCollection(segments,  linewidth=8, cmap=cmap2, norm=norm , array = colors)
            plt.gca().add_collection(lc)


            # Assuming traj is your trajectory data
            first_point = traj.iloc[0]
            last_point = traj.iloc[-1]

            dx = last_point[0] - first_point[0]
            dy = last_point[1] - first_point[1]

            angle_degrees = math.atan2(dy, dx)


            category = categorize_angle(angle_degrees)

            # Remove the axis
            plt.axis('off')

            # Set the figure title as the track ID
            # plt.title(f"Track ID: {trajectoryId}")
            # Show plot
            # plt.show()


            # # Pause for 3 seconds
            # plt.pause(2)

            # # Close the plot
            # plt.close()
            

            plt.savefig(os.path.join(save_dir,f'{category}',f'{trajectoryId}.png'), bbox_inches='tight')
            plt.clf()
        #----------------------------------------------------------------------------
        # STEP 2 : Predict
        #----------------------------------------------------------------------------
        # Convert the figure to a NumPy array
        # fig = plt.gcf()
        # fig.canvas.draw()
        # graph_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        # graph_array = graph_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))


