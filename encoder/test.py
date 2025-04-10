#!/usr/bin/env python3

import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
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
    X_test = np.array([fen_to_tensor(fen) for fen in df['fen']])
    
    # Convert labels to one-hot encoding
    y_test = tf.keras.utils.to_categorical(df['label'] + 1, num_classes=3)  # Convert -1,0,1 to 0,1,2
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test data.
    
    Args:
        model: The trained Keras model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        tuple: (loss, accuracy)
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    return loss, accuracy

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test a CNN model on chess data.')
    parser.add_argument('data_path', type=str, help='Path to the CSV file containing FEN strings and labels')
    args = parser.parse_args()

    # Load the model
    model = tf.keras.models.load_model('final_model.h5')  # Load the trained model

    # Load test data
    X_test, y_test = load_test_data(args.data_path)

    # Evaluate model
    loss, accuracy = evaluate_model(model, X_test, y_test)

    # Print evaluation results
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
