# Aerial-Object-Detection

Object detection on Aerial Images using [DOTA](https://captain-whu.github.io/DOTA/index.html) dataset

# Create the environment

`pip install -r requirements.txt`

Ozstar Supercomputer - `bash environment.sh`

# Yolov5s

Navigate to the Yolov5 directory

`cd yolov5`

Update the path to train and test directories in `yolov5/dota.yaml`

## Train

`python train.py --img <img-size> --batch <batch-size> --epochs 300 --data dota.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt`

## Detect

`python detect.py --source <path-to-test-images> --weights <path-to-yolov5s-best.pt-weights> --conf <default-0.25>`

# Yolov5x

## Train

`python train.py --img <img-size> --batch <batch-size> --epochs 300 --data dota.yaml --cfg models/yolov5x.yaml --weights yolov5x.pt`

## Detect

`python detect.py --source <path-to-test-images> --weights <path-to-yolov5x-best.pt-weights> --conf <default-0.25>`




