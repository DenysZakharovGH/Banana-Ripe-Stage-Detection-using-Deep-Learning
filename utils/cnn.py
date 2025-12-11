from tensorflow.keras import layers, models, metrics
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

from settings import checkpoint_filepath, dir_path, IMAGE_DSIZE


def build_multihead_cnn(input_shape=IMAGE_DSIZE):  # розмір зображення 128х128, 3 канали (RGB)
    model = models.Sequential([
        layers.Conv2D(32, (4, 4), activation='relu', padding='same', input_shape=input_shape),

        # 1 згортковий блок
        layers.Conv2D(64, (4, 4), activation='relu', padding='same'),  # Padding = “доповнення” зображення нулями
        layers.MaxPooling2D((2, 2)),

        # 2 згортковий блок
        layers.Conv2D(128, (4, 4), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.5),

        layers.Flatten(),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='linear')
    ])


    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model


def get_callbacks():
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        #ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True)
    ]



    return callbacks
