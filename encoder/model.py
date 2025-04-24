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


def create_base_model(input_shape=(8, 8, 12), name="base_encoder"):
    """
    Creates the shared base of the CNN model for feature extraction.

    Args:
        input_shape (tuple): The shape of the input tensor (default: (8, 8, 12)).

    Returns:
        tensorflow.keras.models.Model: The base CNN model ending before output heads.
    """
    input_tensor = Input(shape=input_shape, name="input_tensor")

    # --- Shared Convolutional Base ---
    # Block 1
    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(0.001),
        name="conv1",
    )(input_tensor)
    x = BatchNormalization(name="bn1")(x)
    x = Activation("relu", name="relu1")(x)
    x = MaxPooling2D((2, 2), name="pool1")(x)  # Output: (4, 4, 32)

    # Block 2
    x = Conv2D(
        64,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(0.001),
        name="conv2",
    )(x)
    x = BatchNormalization(name="bn2")(x)
    x = Activation("relu", name="relu2")(x)
    x = MaxPooling2D((2, 2), name="pool2")(x)  # Output: (2, 2, 64)

    # Block 3
    x = Conv2D(
        128,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(0.001),
        name="conv3",
    )(x)
    x = BatchNormalization(name="bn3")(x)
    x = Activation("relu", name="relu3")(x)
    x = MaxPooling2D((2, 2), name="pool3")(x)  # Output: (1, 1, 128)

    # --- Shared Dense Layers ---
    x = Flatten(name="flatten")(x)
    x = Dense(128, kernel_regularizer=regularizers.l2(0.001), name="dense_shared")(x)
    x = BatchNormalization(name="bn_dense")(x)
    x = Activation("relu", name="relu_dense")(x)
    # feature vector for the LLM
    shared_output = Dropout(0.4, name="shared_output_dropout")(x)

    # Create the base model
    base_model = Model(
        inputs=input_tensor,
        outputs=shared_output,
        name=name,
    )
    return base_model


def create_cnn_model(base_model=None, learning_rate=0.0005, name="cnn_classifier"):
    """
    Adds classification heads to the base CNN model and compiles it.

    Args:
        base_model (tensorflow.keras.models.Model): The pre-built base CNN model.
        learning_rate (float): The learning rate for the Adam optimizer.

    Returns:
        tensorflow.keras.models.Model: A compiled classifier model.
    """
    # Get the output tensor from the base model
    if not base_model:
        base_model = create_base_model()
    # shared_representation = base_model.output  # Get the output tensor

    # Create fresh input
    input_tensor = Input(shape=base_model.input_shape[1:])

    # Use base_model like a layer
    shared_output = base_model(input_tensor)

    # --- Output Heads ---
    output_material_cmp = Dense(3, activation="softmax", name="output_material_cmp")(
        shared_output
    )

    output_castle_white_kingside = Dense(
        1, activation="sigmoid", name="output_castle_white_kingside"
    )(shared_output)
    output_castle_white_queenside = Dense(
        1, activation="sigmoid", name="output_castle_white_queenside"
    )(shared_output)
    output_castle_black_kingside = Dense(
        1, activation="sigmoid", name="output_castle_black_kingside"
    )(shared_output)
    output_castle_black_queenside = Dense(
        1, activation="sigmoid", name="output_castle_black_queenside"
    )(shared_output)

    # Define the full classifier model
    classifier_model = Model(
        inputs=base_model.input,
        outputs=[
            output_material_cmp,
            output_castle_white_kingside,
            output_castle_white_queenside,
            output_castle_black_kingside,
            output_castle_black_queenside,
        ],
        name=name,
    )

    # Define losses, weights, and metrics
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

    # Compile the model
    classifier_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )

    return classifier_model


if __name__ == "__main__":
    ### Create base model and classifier and print summaries

    # 1. Create the base model / feature extractor
    feature_extractor = create_base_model(input_shape=(8, 8, 12))
    print("Feature extractor summary:")
    feature_extractor.summary()
    print(
        "\nBase Model Output Shape:", feature_extractor.output_shape
    )  # Should be (None, 128)

    # 2. Create the full classifier using the base model
    full_classifier = create_cnn_model(feature_extractor)
    print("\nFull classifier model summary:")
    full_classifier.summary()

    feature_extractor = feature_extractor
    print(f"\nFeature extractor output tensor name: {feature_extractor.output.name}")
    print(f"Feature extractor output shape: {feature_extractor.output_shape}")

    # Load weights into the models
    # full_classifier.load_weights('path/to/full_classifier_weights.h5')
    # feature_extractor.load_weights('path/to/base_model_weights.h5')

    try:
        # Access the base model by its name
        print("Extracting base model by name 'base_encoder'...")
        trained_base_model = full_classifier.get_layer("base_encoder")

        print("Base model extracted successfully.")
        trained_base_model.summary()  # Verify it's the correct model

        # Save the base model separately if needed
        print("Saving extracted base model...")
        trained_base_model.save("trained_base_model.keras")
        print("Base model saved to trained_base_model.keras")

        # Get the weights as a list of numpy arrays
        # base_weights = trained_base_model.get_weights()
        # print(f"Extracted {len(base_weights)} weight arrays from the base model.")

    except ValueError as e:
        print(f"\nError: Could not find a layer named 'base_encoder'. {e}")
        print("Ensure the base model was given this name during creation.")
        print("Consider manual weight transfer.")
        print("\nLayers found in the loaded model:")

        for layer in full_classifier.layers:
            print(f"- {layer.name} (Type: {type(layer).__name__})")
