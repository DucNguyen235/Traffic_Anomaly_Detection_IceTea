# -*- coding: utf-8 -*-
import cv2, os
import numpy as np
from tqdm import tqdm

rt = r'.\data\video'
out_dir = r'.\data\bg'
frame_rate = 30
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
    # Background modeling in forward direction
    for i in videos_id:
        id = str(i)
        # path for background frames
        wrt_bg = out_dir + '/' + f"{id}"
        if not os.path.exists(os.path.join(wrt_bg)):
            os.makedirs(wrt_bg)

        if os.path.exists(os.path.join(rt, id + '.mp4')):
            cap = cv2.VideoCapture(os.path.join(rt, id + '.mp4'))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = tqdm(total=total_frames, unit='frames', desc=f'Extracting frames from {id}')

        ret, frame = cap.read()
        total_pixels = frame.shape[0]*frame.shape[1]
        # build MOG2 model
        # bs = cv2.createBackgroundSubtractorMOG2(120, 16, False)
        bs = cv2.createBackgroundSubtractorMOG2(100, 16, False)
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=0, varThreshold=150, detectShadows=False)

        gray_img1= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bg_og=bg_subtractor.apply(gray_img1)

        count = 0
        is_clear = 0
        while True:
            count += 1
            # print((1.0/frame_rate  * 1000 * count))
            cap.set(cv2.CAP_PROP_POS_MSEC, 1.0/frame_rate  * 1000 * count)
            # cap.set(cv2.CAP_PROP_POS_MSEC, 1.0/frame_rate * 1000 * count)
            # print((1000 * count))

            ret, frame = cap.read()

            
            if not ret:
                break
            fg_mask = bs.apply(frame)
            bg_img = bs.getBackgroundImage()

            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
            # Apply background subtraction on the frame
            fg_mask = bg_subtractor.apply(gray_frame, learningRate=0)

            # Find contours of white regions in the image
            contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Extract the largest contour
            threshold_area = 50
            large_contours = [c for c in contours if cv2.contourArea(c) > threshold_area]

            if (np.sum(fg_mask == 255)/total_pixels < 0.04) and (len(large_contours) < 2):
                is_clear +=1
            else:
                bg_og=bg_subtractor.apply(gray_frame)
                is_clear -=1
            # cv2.imwrite(os.path.join(wrt_bg, 'test_' + id + '_' + str(int(count)).zfill(6) + '.jpg'), bg_img)
            # cv2.imshow("Image with Bounding Box", frame)
            print(is_clear)
            cv2.imshow("bg", bg_img)
            cv2.imshow("fg", bg_og)
            # cv2.imshow("Image with Bounding Box2", frame2)
            if is_clear >= 300: 
                cv2.imwrite(os.path.join(wrt_bg, 'bg_' + id  + '.jpg'), bg_img)
                break
            cv2.waitKey(1)
            # cv2.destroyAllWindows()
            # progress_bar.update(1)
        cap.release()
