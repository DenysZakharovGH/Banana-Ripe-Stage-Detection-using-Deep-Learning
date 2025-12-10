# # Features:
# # 1. Enable a GPU with "cuda"
# # 2. Saving education progress
# # 3. Saving weighs and baises
# # 4. Want to see how looks beetween layers results
#
# # 0. Setup enviroment
# # 0.1. Set up libraries
# # 0.2. Set up folders with images
# # 0.3. Set up global names
#
#
# # 1. About data
# #    1.1 Importing data
# #    1.2 Data destribution with grafic or pie chart # tran vs test
# # 2. Data normalisation and reshape
# #    2.1. images into proper size reshape (might be with cv2 and contours)
# #    2.2. data separation to train and test
#
# # 3. chosing data model and setup
# #     3.1 Batch_size to choose
# #     3.2 Adam parameters
# #     3.3 Dropout
# #     4.1 between education accuracy result
# #   conv 32, 4,4
# #   conv 64, 4,4
# #   conv 128,4,4
# #   conv 128,4,4
# #   512 input Neurons
# #   4 hidden layer
# #   1 output with 0/1 - has tumor or not - softmax activation
#
# # Train CNN
# # show training results, model performance
#
# # Avaluate model
# # confusion matrix
#
# # model results accuracy
#
#
# # zu verbessser
# # es gibt kein Sinn das Bild zu drehen, damit die Zahlen konnen nicht korrekt gelesen werden
import glob
# {
#   "category": "ripe",
#   "edible": true,
#   "days_left": 3,
#   "note": "Найкраще їсти зараз. Через 2–4 дні стане переспілим."
# }

# Категорія	                                                        Чи можна їсти?	                                                                            Орієнтовно скільки ще може лежати (кімнатна температура)
# freshunripe (свіжий зелений, дуже твердий)	                    ❌ Нежелано (не смачний, дуже крохмалистий, але не шкідливий)	 False             5 днів до стиглості,         9 днів до початку псування
# unripe (зелений, але ближче до стиглого)	                        ❌ Не рекомендовано, але безпечно	                             True              3 днів до стиглості,         7 днів до перезрівання
# freshripe (свіжо достиглий, жовтий без плям)	                    ✅ Так, найкращий смак	                                         True              1,                           5 днів до перестигання
# ripe (жовтий, можуть бути дрібні цятки)	                        ✅ Так	                                                         True              0                            2 дні до перестигання
# overripe (багато коричневих плям, дуже м’який)	                ⚠️ Так, але бажано термічно обробити (випічка, смузі)	         True              0                            1 дні, потім швидке псування
# rotten (чорні ділянки, запах, м'якоть слизька або пліснява)                                                                        False             0                            0
#
#

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from settings import train_folder_path, dir_path, checkpoint_filepath, EPOCHS, BATCH_SIZE
from utils.cnn import get_callbacks, build_multihead_cnn
from utils.main_utils import create_train_data, create_train_data_eatable
from utils.save_data_plot import save_data_spread_plot

# download dataset
# get dataset prepared with output_data_structure
# CNN backbone with 3 outputs


# create train data labels and images
train_data_images, train_data_labels = create_train_data_eatable(train_folder_path)

# save data spread image from plt into /docs/
save_data_spread_plot(train_data_labels)

X = train_data_images.astype("float32") / 255.0
Y = train_data_labels


# will keep test data as a path to picture to save memory space
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


model = build_multihead_cnn()
model.summary()


history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[get_callbacks()]
)
model.save(checkpoint_filepath)
# The model weights (that are considered the best) can be loaded as -
#model.load_weights(checkpoint_filepath)

print(history.history.keys())

fig, axes = plt.subplots(ncols=2, figsize=(20, 5))

axes[0].plot(history.history['accuracy'], label='Train accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation accuracy')

axes[0].legend()
#axes[0].title("Accuracy training")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")


axes[1].plot(history.history['val_loss'], label='Validation loss')
axes[1].plot(history.history['loss'], label='loss')
axes[1].legend()
#axes[1].title("loss training")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("loss")

plt.savefig(fr"{dir_path}\docs\data_train.png", dpi=300, bbox_inches="tight")

exit()
