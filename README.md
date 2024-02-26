
### Introduction

#### Anomaly detection for GTEL

Detailed information of NVIDIA AICity Challenge 2021 can be found [here](https://www.aicitychallenge.org/2021-ai-city/).
![overview](data/overview.jpg)
Overview of the architecture of our anomaly detection framework, which consists of background modeling model, training image generator, training model, and the live anomaly detection model.
### Requirements

- Python 3.10
- PyTorch 0.4.1
- Opencv
- sklearn
- Keras
- Ultralytics YOLO v8 


### Installation

1. Install Ultralytics by following this tutorial [official instructions](https://docs.ultralytics.com/quickstart/#install-ultralytics).

### Test video(s)

Since it takes a long time to run this system, we split the task into several steps and provide pretrain results.
1. Run `python ./bg_code/ex_bg_mog.py <video_id>`. Then you will get the video empty background.
2. Run `python ./mask_code/mask_frame_diff.py <video_id>` and `python ./mask_code/mask_track.py <video_id>` to get the masking. Then run `python ./mask_code/mask_fuse.py <video_id>` to combine the two previous masking image.
3. Put the downloaded YOLO model into `./model` (ex: yolov8x), and put pretrain detection into `./model` (ex:vn.pkl).
4. To test a video and show the result.
`python ./det_live.py <video_id> <model_path> <bg_show true/false> <trajectory_show true/false>`


### Train anomaly detector
We use pretrain ResNet50 model from Keras as our feature extractor. We will be training the OC-SVM model in this step and saving its weight.
1. `python ./bg_code/to_csv.py <video_id>` to create tracking csv file. This file will help us to create training images faster.
2. `python ./pred_code/training_image_generator.py <video_id>` to create training images. Choose the all of the normal trajectory images for that road and put it in a folder
3. Configure the script to the video you want to run `python ./model/resnet50_train.py`. After running this file, you should get the model weight for that road type. (.pkl file)











