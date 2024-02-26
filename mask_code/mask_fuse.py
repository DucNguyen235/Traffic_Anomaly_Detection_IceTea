import cv2
import sys
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
        break
    else:
        video_names.append(video_id)
    for video_name in video_names:
        # mask_bg = cv2.imread(fr"./data/mask_diff/" + str(video_name).zfill(3) + ".jpg",0)
        mask_bg = cv2.imread(fr"./data/mask/mask_diff/{video_name:03}.jpg",0)
        kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
        mask_dilate = cv2.erode(mask_bg, kernel_e)
        mask_dilate = cv2.dilate(mask_dilate, kernel)

        mask_dilate[mask_dilate<127] = 0
        mask_dilate[mask_dilate>=127] = 1
        mask_det = cv2.imread(fr"./data/mask/mask_track/mask_{video_name}.png",0)

        # cv2.imshow("Image with Bounding Box", mask_det)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        mask_det[mask_det<127] = 0
        mask_det[mask_det>=127] = 1

        mask_res = (mask_dilate&mask_det)
        mask_res = (mask_dilate|mask_det)

        cv2.imwrite("./data/mask_" + str(video_name).zfill(3) + ".jpg", mask_res*255)
