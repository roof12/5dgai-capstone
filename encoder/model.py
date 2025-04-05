#!/usr/bin/env python3

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_cnn_model():
    """
    Creates a simple CNN model that takes [8, 8, 12] input and outputs 3 classes.
    
    Returns:
        tensorflow.keras.models.Sequential: A compiled CNN model
    """
    model = Sequential([
        # First convolutional layer with padding to maintain dimensions
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(8, 8, 12)),
        MaxPooling2D((2, 2)),
        
        # Second convolutional layer with padding
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Third convolutional layer with padding
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        # Output layer with 3 classes
        Dense(3, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Only run this code if the file is executed directly
if __name__ == "__main__":
    model = create_cnn_model()
    model.summary() 