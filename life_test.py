import cv2
import numpy as np
from keras.src.saving import load_model
from matplotlib import pyplot as plt

from settings import IMAGE_DSIZE, dir_path, checkpoint_filepath

test_image_path = fr"{dir_path}\\data\\live_test\\a921de66-4324-4229-82ba-a68aa941fe8d.jpg"
test_image_path = fr"{dir_path}\\data\\live_test\\8997ff06-79bb-4ee9-9f8a-f08f84cd6ff3.jpg"
model = load_model(checkpoint_filepath)


train_data_images = []
img = cv2.imread(test_image_path)
img_resized = cv2.resize(img, IMAGE_DSIZE[:2])
train_data_images.append(img_resized)


data_images = np.array(train_data_images, dtype=np.float32)
X_test = data_images.astype("float32") / 255.0


y_pred = model.predict(X_test)
y_pred_labels = np.multiply(y_pred,10).squeeze(-1)
pred_time = round(float(y_pred_labels[0] * 24))
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.title(f"Banana will last for: {pred_time} hours", color="green")
plt.imshow(image, cmap="gray")
plt.savefig(fr"{dir_path}\\data\\live_test\\result.png", dpi=300, bbox_inches="tight")