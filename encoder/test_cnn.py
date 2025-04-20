#!/usr/bin/env python3

import numpy as np
import pandas as pd
import tensorflow as tf
from model import create_cnn_model  # Ensure this imports your model correctly
from utils import fen_to_tensor  # Import fen_to_tensor from utils


def load_test_data(data_path):
    """
    Load test data from CSV file containing FEN strings and labels.

    Args:
        data_path (str): Path to the CSV file containing FEN strings and labels

    Returns:
        tuple: (X_test, y_test) numpy arrays
    """
    df = pd.read_csv(data_path)
    X_test = np.array([fen_to_tensor(fen) for fen in df["fen"]])

    # Convert the material comparison label to one-hot encoding (for the 3-class output)
    # Convert -1,0,1 to 0,1,2
    y_test_categorical = tf.keras.utils.to_categorical(
        df["material_cmp"] + 1, num_classes=3
    )

    # Extract the binary labels
    y_test_castle_white_kingside = np.array(df["castle_white_kingside"].astype(int))
    y_test_castle_white_queenside = np.array(df["castle_white_queenside"].astype(int))
    y_test_castle_black_kingside = np.array(df["castle_black_kingside"].astype(int))
    y_test_castle_black_queenside = np.array(df["castle_black_queenside"].astype(int))

    return (
        X_test,
        (
            y_test_categorical,
            y_test_castle_white_kingside,
            y_test_castle_white_queenside,
            y_test_castle_black_kingside,
            y_test_castle_black_queenside,
        ),
    )


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test data.

    Args:
        model: The trained Keras model
        X_test: Test features
        y_test: Test labels (list/tuple of length 5: material_cmp, castling rights)

    Returns:
        dict: A dictionary containing total loss and individual metrics per output.
    """
    results = model.evaluate(X_test, y_test, verbose=1)

    # We should be able to use model.metrics_names. Unfortunately includes
    # the unexpected value 'compile_metrics' and does not include accuracies.
    # As a workaround, it is manually specified here.
    metrics_names = [
        "total_loss",
        "output_material_cmp_loss",
        "output_castle_white_kingside_loss",
        "output_castle_white_queenside_loss",
        "output_castle_black_kingside_loss",
        "output_castle_black_queenside_loss",
        "output_material_cmp_accuracy",
        "output_castle_white_kingside_accuracy",
        "output_castle_white_queenside_accuracy",
        "output_castle_black_kingside_accuracy",
        "output_castle_black_queenside_accuracy",
    ]
    if len(results) != len(metrics_names):
        print(f"Warning: Unexpected number of evaluation results: {len(results)}")

    return dict(zip(metrics_names, results))


def test(data_path):
    # Load the model
    model = tf.keras.models.load_model("best_model.keras")  # Load the trained model

    # Load test data
    X_test, y_test = load_test_data(data_path)

    # Evaluate model
    evaluation_results = evaluate_model(model, X_test, y_test)

    # Print evaluation results
    print(f"Test loss: {evaluation_results['total_loss']:.4f}")
    print(
        f"Test output_material_cmp accuracy: {evaluation_results['output_material_cmp_accuracy']:.4f}"
    )
    print(
        f"Test castle_white_kingside accuracy: {evaluation_results['output_castle_white_kingside_accuracy']:.4f}"
    )
    print(
        f"Test castle_white_queenside accuracy: {evaluation_results['output_castle_white_queenside_accuracy']:.4f}"
    )
    print(
        f"Test castle_black_kingside accuracy: {evaluation_results['output_castle_black_kingside_accuracy']:.4f}"
    )
    print(
        f"Test castle_black_queenside accuracy: {evaluation_results['output_castle_black_queenside_accuracy']:.4f}"
    )
