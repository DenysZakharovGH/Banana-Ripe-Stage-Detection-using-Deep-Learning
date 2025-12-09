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

from settings import train_folder_path, dir_path
from utils.cnn import get_callbacks, build_multihead_cnn
from utils.main_utils import create_train_data
from utils.save_data_plot import save_data_spread_plot

# download dataset
# get dataset prepared with output_data_structure
# CNN backbone with 3 outputs


# create train data labels and images
train_data_images, train_data_labels = create_train_data(train_folder_path)

# save data spread image from plt into /docs/
save_data_spread_plot(train_data_labels)

# 1️⃣ Перемістимо осі, щоб перший індекс був кількістю зразків
X = np.moveaxis(train_data_images, [0, 1, 2, 3, 4], [3, 1, 2, 0, 4])
Y = train_data_labels

# will keep test data as a path to picture to save memory space
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

model = build_multihead_cnn()
model.summary()


history = model.fit(
    X_train,
    {
        "bin_out": y_train[:, 0],  # 0 або 1
        "reg1_out": y_train[:, 1],  # число
        "reg2_out": y_train[:, 2]  # число
    },
    epochs=1,
    batch_size=32,
    callbacks=[get_callbacks()]
)

print(history.history.keys())

fig, axes = plt.subplots(ncols=3, figsize=(25, 6))

# ---------------------------------------------------------
# 1) Accuracy (лише для бінарної голови)
# ---------------------------------------------------------
axes[0].plot(history.history['bin_out_accuracy'], label='Train bin_acc')
axes[0].set_title("Binary Head Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

# ---------------------------------------------------------
# 2) MAE для регресійних виходів
# ---------------------------------------------------------
axes[1].plot(history.history['reg1_out_mae'], label='Train reg1_mae')
#axes[1].plot(history.history['val_reg1_out_mae'], label='Val reg1_mae')

axes[1].plot(history.history['reg2_out_mae'], label='Train reg2_mae')
#axes[1].plot(history.history['val_reg2_out_mae'], label='Val reg2_mae')

axes[1].set_title("Regression Heads MAE")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MAE")
axes[1].legend()

# ---------------------------------------------------------
# 3) Загальний loss
# ---------------------------------------------------------
axes[2].plot(history.history['loss'], label='Train loss')
#axes[2].plot(history.history['val_loss'], label='Val loss')

axes[2].set_title("Total Loss")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Loss")
axes[2].legend()

plt.tight_layout()
plt.savefig(fr"{dir_path}\docs\data_train.png", dpi=300, bbox_inches="tight")
plt.show()


print(X_train[0])
print(y_train[0])

exit()
