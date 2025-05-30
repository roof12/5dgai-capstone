#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import create_cnn_model
import pandas as pd
from utils import fen_to_tensor


def load_data(data_path):
    """
    Load training data from CSV file containing FEN strings and labels.

    Args:
        data_path (str): Path to the CSV file containing FEN strings and labels

    Returns:
        tuple: (X_train, y_train) numpy arrays
    """
    # Read CSV file
    df = pd.read_csv(data_path)

    # Convert FEN strings to tensors
    X_train = np.array([fen_to_tensor(fen) for fen in df["fen"]])

    # Convert the material comparison label to one-hot encoding (for the 3-class output)
    # Convert -1,0,1 to 0,1,2
    y_train_categorical = tf.keras.utils.to_categorical(
        df["material_cmp"] + 1, num_classes=3
    )

    # Extract the binary labels
    y_train_castle_white_kingside = np.array(df["castle_white_kingside"].astype(int))
    y_train_castle_white_queenside = np.array(df["castle_white_queenside"].astype(int))
    y_train_castle_black_kingside = np.array(df["castle_black_kingside"].astype(int))
    y_train_castle_black_queenside = np.array(df["castle_black_queenside"].astype(int))

    return (
        X_train,
        (
            y_train_categorical,
            y_train_castle_white_kingside,
            y_train_castle_white_queenside,
            y_train_castle_black_kingside,
            y_train_castle_black_queenside,
        ),
    )


def train_model(
    model, X_train, y_train, batch_size=32, epochs=50, validation_split=0.2
):
    """
    Train the model with the given data.

    Args:
        model: The compiled Keras model
        X_train: Training features
        y_train: Training labels
        batch_size: Batch size for training
        epochs: Number of epochs to train
        validation_split: Fraction of data to use for validation

    Returns:
        History object containing training metrics
    """
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            "best_model.keras",
            monitor="val_output_material_cmp_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_output_material_cmp_loss",
            patience=5,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
    ]

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=0,
    )

    return history


def train(data_path, batch_size=32, epochs=50, validation_split=0.2):
    # Create model
    model = create_cnn_model()

    # Load data
    X_train, y_train = load_data(data_path)

    # Train model
    history = train_model(model, X_train, y_train, batch_size, epochs)

    # Save final model
    model.save("final_model.keras")

    # Print training summary
    print("\nTraining completed!")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(
        f"Final validation accuracy output_material_cmp: {history.history['val_output_material_cmp_accuracy'][-1]:.4f}"
    )
    print(
        f"Final validation accuracy castle_white_kingside: {history.history['val_output_castle_white_kingside_accuracy'][-1]:.4f}"
    )
    print(
        f"Final validation accuracy castle_white_queenside: {history.history['val_output_castle_white_queenside_accuracy'][-1]:.4f}"
    )
    print(
        f"Final validation accuracy castle_black_kingside: {history.history['val_output_castle_black_kingside_accuracy'][-1]:.4f}"
    )
    print(
        f"Final validation accuracy castle_black_queenside: {history.history['val_output_castle_black_queenside_accuracy'][-1]:.4f}"
    )

