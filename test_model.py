import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model
from tensorflow.python.ops.metrics_impl import precision

from settings import checkpoint_filepath, test_folder_path, dir_path
from utils.main_utils import create_train_data, create_train_data_days_left
from utils.save_data_plot import save_data_spread_plot


counter = 0
colomn = 5
rows = 6
days_tolerance = 1

# create train data labels and images
train_data_images, train_data_labels = create_train_data_days_left(test_folder_path)


# 1️⃣ Перемістимо осі, щоб перший індекс був кількістю зразків
X_test = train_data_images.astype("float32") / 255.0
y_test = train_data_labels

model = load_model(checkpoint_filepath)

y_pred = model.predict(X_test)
y_pred_labels = np.multiply(y_pred,10).squeeze(-1)

plt.figure(figsize=(15, 10))
for i in random.sample(range(0, len(y_pred_labels)), rows*colomn):
    plt.subplot(colomn, rows, counter + 1)

    image = X_test[i]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image, cmap="gray")
    plt.axis("off")

    true_label = y_test[i].astype(int)
    pred_label = round(float(y_pred_labels[i]), 2)

    color = "green" if abs(true_label - pred_label) < days_tolerance else "red"
    plt.title(f"Pred days left: {pred_label}\nTrue days left:: {true_label}", color=color)

    counter +=1
    if counter >= rows*colomn: break

plt.tight_layout()

plt.savefig(fr"{dir_path}\docs\result_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

