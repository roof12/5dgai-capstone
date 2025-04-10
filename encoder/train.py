#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import create_cnn_model
import os
import pandas as pd
import chess
import argparse  # Import argparse
from utils import fen_to_tensor  # Import fen_to_tensor from utils

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
    X_train = np.array([fen_to_tensor(fen) for fen in df['fen']])
    
    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(df['label'] + 1, num_classes=3)  # Convert -1,0,1 to 0,1,2
    
    return X_train, y_train

def train_model(model, X_train, y_train, batch_size=32, epochs=50, validation_split=0.2):
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
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a CNN model on chess data.')
    parser.add_argument('data_path', type=str, help='Path to the CSV file containing FEN strings and labels')
    args = parser.parse_args()  # Parse the arguments

    # Create model
    model = create_cnn_model()
    
    # Load data
    X_train, y_train = load_data(args.data_path)  # Use the command line argument for data_path
    
    # Train model
    history = train_model(
        model,
        X_train,
        y_train,
        batch_size=32,
        epochs=50
    )
    
    # Save final model
    model.save('final_model.h5')
    
    # Print training summary
    print("\nTraining completed!")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

def main_check():
    # Load one row of data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'medium.csv')
    df = pd.read_csv(data_path)
    sample_fen = df['fen'].iloc[0]
    
    # Convert to tensor
    tensor = fen_to_tensor(sample_fen)
 
    #  tensor = np.moveaxis(tensor, -1, 0)
    for channel in range(tensor.shape[2]):
        print(f"Channel {channel}:")
        print(tensor[:, :, channel])
        print("\n")
    
    # Print results
    print(f"Sample FEN: {sample_fen}")
    print("\nTensor shape:", tensor.shape)
    print("\nTensor representation:")

if __name__ == "__main__":
    main()
