"""
Program to split the download images of DOTA dataset to train and val
"""

import os
import shutil

DATA_DIR = "/fred/oz138/COS80028/P2/rakesh/data/"


def split_data(type='train'):
    label_dir = "/labelTxt"
    image_labels = os.listdir(DATA_DIR + type + label_dir)
    for element in image_labels:
        image_id = element.split(".txt")[0]
        image = image_id + ".png"
        shutil.copy(DATA_DIR+"images/"+image, DATA_DIR+type+"/images/"+image)


if __name__ == "__main__":
    split_data(train)
    split_data(val)