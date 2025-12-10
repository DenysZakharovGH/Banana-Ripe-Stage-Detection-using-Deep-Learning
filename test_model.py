import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model

from settings import checkpoint_filepath, test_folder_path
from utils.main_utils import create_train_data
from utils.save_data_plot import save_data_spread_plot

# create train data labels and images
train_data_images, train_data_labels = create_train_data(test_folder_path)


# 1️⃣ Перемістимо осі, щоб перший індекс був кількістю зразків
X_test = train_data_images.astype("float32") / 255.0
y_test = train_data_labels

model = load_model(checkpoint_filepath)

results = model.predict(X_test)

# Remove the last singleton dimension
arr_squeezed = np.squeeze(results)  # shape becomes (3, 562)

# Transpose to swap axes
results_final = arr_squeezed.T      # shape becomes (562, 3)

print(np.asarray(results_final).shape)
print(np.asarray(y_test).shape)


for iter in range(100, 400, 1):

    print("Predicted values",results_final[iter])
    print("Real      values",y_test[iter])
    image_np = X_test[iter]
    cv2.imshow("test", cv2.resize(image_np, (512,512)) )
    cv2.waitKey(0)



outputs = ["bin_out", "reg1_out", "reg2_out"]
y_test_dict = {
    "bin_out": y_test[:, 0],
    "reg1_out": y_test[:, 1],
    "reg2_out": y_test[:, 2]
}

results = model.evaluate(X_test, y_test_dict, verbose=1)
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")



