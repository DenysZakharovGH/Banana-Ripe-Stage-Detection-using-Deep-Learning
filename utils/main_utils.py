import glob

import cv2
import numpy as np

from data.data_structure import output_data_structure
from settings import IMAGE_DSIZE, TRAIN_SIZE
from typing import Tuple


def create_train_data_eatable(folder_path: str = "") -> Tuple[np.array, np.array]:
    train_data_images = []
    train_data_labels = []
    for key in output_data_structure.keys():
        folder_path_local = fr"{folder_path}\\{key}\\*.*"
        counter = 0
        for filename in glob.glob(folder_path_local):
            img_origin = cv2.imread(filename)
            img_resized = cv2.resize(img_origin, IMAGE_DSIZE[:2])
            train_data_images.append(img_resized)
            train_data_labels.append(output_data_structure[key]["eatable"])
            counter += 1
            if counter == TRAIN_SIZE:
                break

    return np.array(train_data_images, dtype=np.float32), np.array(train_data_labels)


def create_train_data(folder_path: str = "") -> Tuple[np.array, np.array]:
    train_data_images = []
    train_data_labels = []
    for key in output_data_structure.keys():
        folder_path_local = fr"{folder_path}\\{key}\\*.*"
        counter = 0
        for filename in glob.glob(folder_path_local):
            img_origin = cv2.imread(filename)
            img_resized = cv2.resize(img_origin, IMAGE_DSIZE[:2])
            train_data_images.append(img_resized)
            train_data_labels.append(list(output_data_structure[key]["days_left"]))
            counter += 1
            if counter == TRAIN_SIZE:
                break

    #return np.array([train_data_images]).T, np.array(train_data_labels)

    return np.array(train_data_images, dtype=np.float32), np.array(train_data_labels)