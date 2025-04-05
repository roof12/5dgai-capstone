#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import create_cnn_model
import os
import pandas as pd
import chess

def fen_to_tensor(fen):
    """
    Convert a FEN string to a [8, 8, 12] tensor representation.
    Each channel represents a different piece type and color.
    Channel order: white pawns, knights, bishops, rooks, queens, kings,
                  black pawns, knights, bishops, rooks, queens, kings
    
    Args:
        fen (str): FEN string representing a chess position
        
    Returns:
        numpy.ndarray: [8, 8, 12] tensor representation of the board
    """
    board = chess.Board(fen)
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    
    # Map piece types to channel indices
    piece_to_channel = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11
    }
    
    # Fill the tensor
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            row = 7 - (square // 8)  # Convert from chess notation to array indices
            col = square % 8
            channel = piece_to_channel[(piece.piece_type, piece.color)]
            tensor[row, col, channel] = 1.0
            
    return tensor

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
    # Create model
    model = create_cnn_model()
    
    # Load data
    data_path = 'data/out.csv'  # Update this path to your data directory
    X_train, y_train = load_data(data_path)
    
    # Train model
    history = train_model(
        model,
        X_train,
        y_train,
        batch_size=64,
        epochs=50
    )
    
    # Save final model
    model.save('final_model.h5')
    
    # Print training summary
    print("\nTraining completed!")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

if __name__ == "__main__":
    main()
