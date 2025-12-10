import os

dir_path = os.getcwd()
test_folder_path = fr"{dir_path}\\data\\test"
train_folder_path = fr"{dir_path}\\data\\train"
valid_folder_path = fr"{dir_path}\\data\\valid"
IMAGE_DSIZE = (128, 128, 3)
TRAIN_SIZE = 100000

EPOCHS = 30
BATCH_SIZE = 32
checkpoint_filepath = fr"{dir_path}\\models\\checkpoint_weights.keras"