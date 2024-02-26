import cv2
import os
import numpy as np
import sys
import skimage
from skimage.measure import label 
from scipy.ndimage.filters import gaussian_filter
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
    for video in videos_id:
        if True:
            count = 0
            out = 0 
            #read video
            # cap = cv2.VideoCapture(os.path.join(rt, "\\"+video ))
            cap = cv2.VideoCapture(dir_path+ fr"\{video}.mp4" )
            ret, frame = cap.read()
            
            while ret:
                # if count % 5 == 0:
                last_frame = frame
                count += 1
                cap.set(cv2.CAP_PROP_POS_MSEC, 0.2 * 1000 * count)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                fg = cv2.subtract(frame,last_frame)
                fg = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
                _, fg1 = cv2.threshold(fg, 100, 255, cv2.THRESH_BINARY)
                fg1[fg1==255] = 1
                
                if sum(sum(fg1)) > 13000:
                    continue
                
                out = cv2.bitwise_or(out,fg) #||
                
                out = cv2.medianBlur(out, 3) 
                
                out = cv2.GaussianBlur(out, (3, 3), 0) 
                
                _, out = cv2.threshold(out, 99, 255, cv2.THRESH_BINARY)
                
                
            min_area = 10000   
            mask = label(out, connectivity = 1)
            num = np.max(mask)
            for i in range(1,int(num+1)):
                if np.sum(mask==i)<min_area:
                    mask[mask==i]=0     
            mask = mask>0
            mask = mask.astype(float)
            dirname=os.path.dirname
            wrt_fgmask = os.path.join(dirname(dirname(__file__)), os.path.join('data\mask\mask_diff'))
            cv2.imwrite(os.path.join(wrt_fgmask, str(video).zfill(3) + '.jpg'), mask*255)


