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

dir_path = os.getcwd()
test_folder_path = fr"{dir_path}\\data\\test"
train_folder_path = fr"{dir_path}\\data\\train"
valid_folder_path = fr"{dir_path}\\data\\valid"


output_data_structure = {
    "fresh_unripe":{"eatable": 0, "days_maturity":5, "days_left":9},
    "unripe":      {"eatable": 1, "days_maturity":3, "days_left":7},
    "fresh_ripe":  {"eatable": 1, "days_maturity":1, "days_left":5},
    "ripe":        {"eatable": 1, "days_maturity":0, "days_left":2.5},
    "overripe":    {"eatable": 1, "days_maturity":0, "days_left":1},
    "rotten":      {"eatable": 0, "days_maturity":0, "days_left":0},
 }

# download dataset
# get dataset prepared with output_data_structure
# CNN backbone with 3 outputs

