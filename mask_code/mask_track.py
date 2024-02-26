#!coding: utf-8
import numpy as np
import cv2 
import os
from skimage.measure import label
from scipy.ndimage.filters import gaussian_filter
import pickle
import pdb
from numpy import zeros, ones,empty
import sys
from ultralytics import YOLO

eps = 1.0E-7

import numpy as np
import math
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Image generator from CSV")
    parser.add_argument("video_id", default= "run_all_file", type=str, help="Video ID default will run all video in the folder")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    video_id = args.video_id
    videos_id = []
    dir_path = r"..\data\video"

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
        videos_id.append(video_id)
    model = YOLO("yolov8x.pt")

    for video in videos_id:
        cap = cv2.VideoCapture(str(video_id) + '.mp4')
        ret, frame = cap.read()
        im = frame

        frame_rate = 0
        dict_track = {}
        while ret:
            #----------------------------------------------------------------------------
            #READ AND RECORD BOUNDING BOX FROM VIDEO
            #----------------------------------------------------------------------------
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame,(1280,720))
            results = model.track(frame,show = False, device = 0,verbose= False, persist= True)
            results = model.track(frame, persist=True, verbose=False)
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            if results[0].boxes.id is None:
                continue
            track_ids = results[0].boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                x1, y1, w, h = box
                center_x = x1 + w / 2
                center_y = y1 + h / 2
                if track_id not in dict_track:
                    dict_track[track_id] = []
                    dict_track[track_id].append([frame_rate,center_x,center_y,[int(x1),int(y1),int(x1+w),int(y1+h)]])
                else:
                    dict_track[track_id].append([frame_rate,center_x,center_y,[int(x1),int(y1),int(x1+w),int(y1+h)]])
            frame_rate += 1
            # Create a dictionary for the current frame's data

        h,w,c = im.shape
        mat = np.zeros((h,w))
        for car_id in dict_track.keys():
            if len(dict_track[car_id]) < 5:
                continue
            total_num = len(dict_track[car_id])
            P0 = [dict_track[car_id][0][1], dict_track[car_id][0][2]]
            Pn = [dict_track[car_id][-1][1], dict_track[car_id][-1][2]]
            
            if math.sqrt((P0[0]-Pn[0])**2 + (P0[1]-Pn[1])**2) < 8 and Pn[1] < 100:
                continue
            if math.sqrt((P0[0]-Pn[0])**2 + (P0[1]-Pn[1])**2) < 50 and Pn[1] >= 100:
                continue
            
            green = (0, 255, 0) #4
            #cv2.line(im, (P0[0], P0[1]), (Pn[0], Pn[1]), green) #5
            
            for box in dict_track[car_id]:
                h = box[3][3] - box[3][1]
                w = box[3][2] - box[3][0]
                
                if w > 50 or h > 50:
                    mat[box[3][1]+int(h/4.0-1):box[3][3]-int(h/4.0-1),box[3][0]+int(w/4.0-1):box[3][2]-int(w/4.0-1)] += 1
                else:
                    mat[box[3][1]:box[3][3],box[3][0]:box[3][2]] += 1
                

        min_area = 200

        mask= mat>0 

        mask = label(mask, connectivity = 1)
        num = np.max(mask)

        for i in range(1,int(num+1)):
            if np.sum(mask==i)<min_area:
                mask[mask==i]=0     
        mask = mask>0
        mask = mask.astype(float)

        kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        mask = cv2.erode(mask, kernel_e)
        mask = mask.astype(np.uint8)

        mask[mask==1]=255
        contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) < 3000:
                cv2.fillConvexPoly(mask, contours[i], 0)
                                

        mask_png = np.zeros(mask.shape)
        mask_png[mask==255] = 1
        mask_png[mask==0] = 0

        if not os.path.exists("./data/mask_track/"):
            os.makedirs("./data/mask_track/")
        cv2.imwrite("./data/mask_track/mask_%s.png"%str(1),mask_png*255)



                
            
                
