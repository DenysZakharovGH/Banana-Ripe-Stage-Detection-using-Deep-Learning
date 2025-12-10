import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model

from settings import checkpoint_filepath, test_folder_path, dir_path
from utils.main_utils import create_train_data, create_train_data_eatable
from utils.save_data_plot import save_data_spread_plot

# create train data labels and images
train_data_images, train_data_labels = create_train_data_eatable(test_folder_path)


# 1️⃣ Перемістимо осі, щоб перший індекс був кількістю зразків
X_test = train_data_images.astype("float32") / 255.0
y_test = train_data_labels

model = load_model(checkpoint_filepath)

y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)

# confusion matrix plot
cm = confusion_matrix(y_test, y_pred_labels, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Eatable", "NOT Eatable"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(fr"{dir_path}\docs\confusion_matrix.png", dpi=300, bbox_inches="tight")

print("Accuracy:", accuracy_score(y_test, y_pred_labels))
print("Classification report:\n", classification_report(y_test, y_pred_labels))

incorrect = np.where(y_pred_labels.flatten() != y_test.flatten())[0]

# for i in incorrect:
#     test_image = X_test[i]
#     plt.imshow(test_image)
#     plt.title(f"True: {y_test[i]}, Predicted: {y_pred_labels[i][0]}")
#     plt.show()



counter = 0
colomn = 5
rows = 6

import random


plt.figure(figsize=(15, 10))
for i in random.sample(range(0, len(y_pred_labels)), rows*colomn):
    plt.subplot(colomn, rows, counter + 1)

    image = X_test[i]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image, cmap="gray")
    plt.axis("off")

    true_label = y_test[i].astype(int)
    pred_label = y_pred_labels[i]

    color = "green" if true_label == pred_label else "red"
    plt.title(f"Pred: {pred_label[0]}\nTrue: {true_label}", color=color)

    counter +=1
    if counter >= rows*colomn: break


plt.tight_layout()

plt.savefig(fr"{dir_path}\docs\result_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

