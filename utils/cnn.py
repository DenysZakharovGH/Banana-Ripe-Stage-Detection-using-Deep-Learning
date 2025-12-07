from tensorflow.keras import layers, models
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

from settings import dir_path


def build_multihead_cnn(input_shape=(128, 128, 3)):  # розмір зображення 128х128, 3 канали (RGB)

    inputs = layers.Input(shape=input_shape)

    # Backbone
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Output head with 3 values

    # --- Голова 1 (бінарна класифікація) ---
    head1 = layers.Dense(1, activation='sigmoid', name='bin_out')(x)

    # --- Голова 2 (регресія) ---
    head2 = layers.Dense(1, activation='linear', name='reg1_out')(x)

    # --- Голова 3 (регресія) ---
    head3 = layers.Dense(1, activation='linear', name='reg2_out')(x)

    model = models.Model(inputs, [head1, head2, head3] )

    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.compile(
        optimizer="adam",
        loss={
            "bin_out": "binary_crossentropy",
            "reg1_out": "mse",
            "reg2_out": "mse"
        },
        loss_weights={
            "bin_out": 1.0,
            "reg1_out": 0.5,
            "reg2_out": 0.5
        },
        metrics={
            "bin_out": ["accuracy"],
            "reg1_out": ["mae"],
            "reg2_out": ["mae"]
        }
    )

    return model


def get_callbacks():
    callbacks = [
        EarlyStopping(monitor='reg1_out_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(fr"{dir_path}\\models\\best_model.keras", monitor='accuracy', save_best_only=True)
    ]
    return callbacks
