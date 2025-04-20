#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


def create_cnn_model():
    """
    Creates a simple CNN model that takes [8, 8, 12] input and outputs 3 classes.

    Returns:
        tensorflow.keras.models.Sequential: A compiled CNN model
    """

    # Define Input Shape
    input_shape = (8, 8, 12)
    input_tensor = Input(shape=input_shape)

    # --- Shared Convolutional Base ---
    # Block 1
    x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001))(
        input_tensor
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)  # Output: (4, 4, 32)

    # Block 2
    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)  # Output: (2, 2, 64)

    # Block 3
    # NOTE: Consider removing this pooling layer if preserving spatial info is critical
    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001))(
        x
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)  # Output: (1, 1, 128)

    # --- Shared Dense Layers ---
    x = Flatten()(x)
    x = Dense(128, kernel_regularizer=regularizers.l2(0.001))(x)  #
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    shared_representation = Dropout(0.4)(x)  # Output before branching

    # --- Output Heads ---
    output_material_cmp = Dense(3, activation="softmax", name="output_material_cmp")(
        shared_representation
    )

    output_castle_white_kingside = Dense(
        1, activation="sigmoid", name="output_castle_white_kingside"
    )(shared_representation)
    output_castle_white_queenside = Dense(
        1, activation="sigmoid", name="output_castle_white_queenside"
    )(shared_representation)
    output_castle_black_kingside = Dense(
        1, activation="sigmoid", name="output_castle_black_kingside"
    )(shared_representation)
    output_castle_black_queenside = Dense(
        1, activation="sigmoid", name="output_castle_black_queenside"
    )(shared_representation)

    # The model takes the input_tensor and outputs a list of the defined output layers
    model = Model(
        inputs=input_tensor,
        outputs=[
            output_material_cmp,
            output_castle_white_kingside,
            output_castle_white_queenside,
            output_castle_black_kingside,
            output_castle_black_queenside,
        ],
    )

    # Compile the model
    losses = {
        "output_material_cmp": "categorical_crossentropy",
        "output_castle_white_kingside": "binary_crossentropy",
        "output_castle_white_queenside": "binary_crossentropy",
        "output_castle_black_kingside": "binary_crossentropy",
        "output_castle_black_queenside": "binary_crossentropy",
    }

    loss_weights = {
        "output_material_cmp": 1.0,
        "output_castle_white_kingside": 1.0,
        "output_castle_white_queenside": 1.0,
        "output_castle_black_kingside": 1.0,
        "output_castle_black_queenside": 1.0,
    }

    metrics = {
        "output_material_cmp": ["accuracy"],
        "output_castle_white_kingside": ["accuracy"],
        "output_castle_white_queenside": ["accuracy"],
        "output_castle_black_kingside": ["accuracy"],
        "output_castle_black_queenside": ["accuracy"],
    }

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    return model


if __name__ == "__main__":
    model = create_cnn_model()
    model.summary()
