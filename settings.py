import os

dir_path = os.getcwd()
test_folder_path = fr"{dir_path}\\data\\test"
train_folder_path = fr"{dir_path}\\data\\train"
valid_folder_path = fr"{dir_path}\\data\\valid"
IMAGE_DSIZE = (128, 128)
TRAIN_SIZE = 100000