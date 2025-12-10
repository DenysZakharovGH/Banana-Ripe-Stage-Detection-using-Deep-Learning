from tensorflow.keras import layers, models, metrics
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

from settings import checkpoint_filepath, dir_path, IMAGE_DSIZE


def build_multihead_cnn(input_shape=IMAGE_DSIZE):  # розмір зображення 128х128, 3 канали (RGB)

    # inputs = layers.Input(shape=input_shape)
    #
    # # Backbone
    # x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    # x = layers.MaxPooling2D()(x)
    #
    # x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    # x = layers.MaxPooling2D()(x)
    #
    # x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    # x = layers.GlobalAveragePooling2D()(x)
    #
    # # Output head with 3 values
    #
    # # --- Голова 1 (бінарна класифікація) ---
    # head1 = layers.Dense(1, activation='sigmoid', name='bin_out')(x)
    #
    # # --- Голова 2 (регресія) ---
    # head2 = layers.Dense(1, activation='linear', name='reg1_out')(x)
    #
    # # --- Голова 3 (регресія) ---
    # head3 = layers.Dense(1, activation='linear', name='reg2_out')(x)
    #
    # model = models.Model(inputs, [head1, head2, head3] )

    inputs = layers.Input(shape=input_shape)

    x = layers.Rescaling(1 / 255)(inputs)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Shared MLP
    shared = layers.Dense(256, activation='relu')(x)
    shared = layers.Dropout(0.3)(shared)

    # Heads
    head1 = layers.Dense(128, activation='relu')(shared)
    head1 = layers.Dense(1, activation='sigmoid', name='bin_out')(head1)

    head2 = layers.Dense(128, activation='relu')(shared)
    head2 = layers.Dense(1, activation='linear', name='reg1_out')(head2)

    head3 = layers.Dense(128, activation='relu')(shared)
    head3 = layers.Dense(1, activation='linear', name='reg2_out')(head3)

    model = models.Model(inputs, [head1, head2, head3])

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
            "bin_out": ["accuracy", metrics.AUC()],
            "reg1_out": ["mae"],
            "reg2_out": ["mae"]
        }
    )


    return model


def get_callbacks():
    # callbacks1 = [
    #     EarlyStopping(monitor='val_bin_out_accuracy', patience=5, restore_best_weights=True),
    #     ModelCheckpoint(fr"{dir_path}\\models\\best.h5", monitor='val_loss', save_best_only=True)
    # ]
    # Callback — це “автоматичний наглядач” за тренуванням моделі.
    # Він постійно моніторить метрики (accuracy, loss, mae, auc і т. д.) і виконує певні дії.
    callbacks = [
        EarlyStopping(
            monitor="val_bin_out_accuracy",  # monitor head-specific metric
            patience=5,
            mode="max",
            restore_best_weights=True
        ),

        # ModelCheckpoint(
        #     filepath=checkpoint_filepath,
        #     monitor="val_loss",  # or choose a specific head metric
        #     save_best_only=True,
        #     mode="min"
        # ),
    ]
    return callbacks
