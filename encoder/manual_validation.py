#!/usr/bin/env python3

# This script is used for manual validation of test results.
# It lists the input FEN and labels, and the predicted values.

from utils import fen_to_tensor  # Import fen_to_tensor from utils
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import os


# --- Function to load and evaluate ---
def evaluate_csv_per_row(model_path, csv_path):
    """
    Loads a saved model, evaluates it on data from a CSV,
    and reports per-row and overall accuracies for each output.

    Args:
        model_path (str): Path to the saved Keras model file (.keras or .h5).
        csv_path (str): Path to the CSV data file.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    if not os.path.exists(csv_path):
        print(f"Error: Data CSV file not found at {csv_path}")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    # If you used any custom objects (layers, loss functions, metrics)
    # in your model definition, you'll need to pass them to custom_objects
    # in load_model. Example: custom_objects={'CustomLayer': CustomLayer}
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")

    # --- Prepare Input Data (X) ---
    print("Converting FEN strings to tensors...")
    X = np.array([fen_to_tensor(fen) for fen in df["fen"]])
    print(f"Input shape: {X.shape}")

    # --- Prepare True Labels (y_true) ---
    # Ensure column names match your CSV and the order matches your model's outputs
    # Assuming order: categorical, wk_binary, wq_binary, bk_binary, bq_binary
    label_columns = [
        "material_cmp",  # Categorical (-1, 0, 1) -> convert to (0, 1, 2)
        "castle_white_kingside",  # Binary (0, 1)
        "castle_white_queenside",  # Binary (0, 1)
        "castle_black_kingside",  # Binary (0, 1)
        "castle_black_queenside",  # Binary (0, 1) - **Adjust column name if needed**
    ]

    # Check if all expected label columns exist
    if not all(col in df.columns for col in label_columns):
        missing = [col for col in label_columns if col not in df.columns]
        print(f"Error: Missing expected label columns in CSV: {missing}")
        print(f"Please ensure your CSV contains columns: {label_columns}")
        sys.exit(1)

    print("Preparing true labels...")
    y_true = [
        np.array(df[label_columns[0]] + 1),  # Categorical: Convert -1,0,1 to 0,1,2
        np.array(df[label_columns[1]]),  # Binary 1 (wk)
        np.array(df[label_columns[2]]),  # Binary 2 (wq)
        np.array(df[label_columns[3]]),  # Binary 3 (bk)
        np.array(df[label_columns[4]]),  # Binary 4 (bq)
    ]
    print(f"Prepared {len(y_true)} true label arrays.")
    # print(f"Example true labels (row 0): {[arr[0] for arr in y_true]}") # Optional check

    # --- Make Predictions ---
    print("Making predictions...")
    # model.predict returns a list of arrays, one for each output
    predictions = model.predict(X)
    print(f"Received {len(predictions)} prediction arrays.")
    # print(f"Example prediction shapes: {[p.shape for p in predictions]}") # Optional check

    # --- Process Predictions to Get Predicted Labels ---
    print("Processing predictions...")
    predicted_labels = []

    # Output 0: Categorical (Softmax output)
    # Get the class index with the highest probability
    predicted_labels.append(np.argmax(predictions[0], axis=1))
    # print(f"Example pred_cat_idx (row 0): {predicted_labels[0][0]}") # Optional check

    # Outputs 1-4: Binary (Sigmoid output)
    # Round the sigmoid output to 0 or 1
    for i in range(1, 5):
        # Ensure prediction shape is handled (might be (N, 1) or (N,))
        pred_binary = (
            predictions[i].flatten() if predictions[i].ndim > 1 else predictions[i]
        )
        predicted_labels.append((pred_binary > 0.5).astype(int))
        # print(f"Example pred_bin{i} (row 0): {predicted_labels[i][0]}") # Optional check

    # --- Compare True Labels and Predictions & Calculate Accuracies ---
    print("\n--- Evaluation Results ---")
    num_samples = len(df)
    overall_correct_counts = np.zeros(5, dtype=int)  # Counter for each of the 5 outputs

    # Lists to store accuracy results per row
    row_accuracies = []
    output_match_statuses = [
        [] for _ in range(5)
    ]  # Store True/False for each output per row

    print(
        "Row | True Labels | Predicted Labels | Match Statuses      | Row Accuracy | FEN"
    )
    print("-" * 120)

    for i in range(num_samples):
        true_row = [arr[i] for arr in y_true]
        pred_row = [arr[i] for arr in predicted_labels]

        # Check if each output prediction matches the true label for this row
        matches_row = [(true_row[j] == pred_row[j]) for j in range(5)]

        # Count how many outputs were correct in this row
        correct_in_row = sum(matches_row)

        # Calculate per-row accuracy
        accuracy_row = correct_in_row / 5.0

        # Update overall correct counts for each output
        for j in range(5):
            if matches_row[j]:
                overall_correct_counts[j] += 1

        # Store match statuses for potential later analysis (optional)
        for j in range(5):
            output_match_statuses[j].append(matches_row[j])

        row_accuracies.append(accuracy_row)

        # Print results for the current row
        # Using f-string formatting with padding for columns
        print(
            f"{i:<3} | {true_row!s:<11} | {pred_row!s:<16} | {matches_row!s:<19} | {accuracy_row:<12.4f} | {df['fen'][i]}"
        )  # Added FEN

    print("-" * 120)

    # --- Report Overall Accuracies ---
    print("\n--- Overall Output Accuracies ---")
    output_names = [
        "Material Comparison (3-class)",
        "Castle White Kingside (Binary)",
        "Castle White Queenside (Binary)",
        "Castle Black Kingside (Binary)",
        "Castle Black Queenside (Binary)",  # Matches the 5 outputs
    ]

    overall_accuracies = [(count / num_samples) for count in overall_correct_counts]

    for i in range(5):
        print(f"{output_names[i]}: {overall_accuracies[i]:.4f}")

    # Optional: Calculate average per-row accuracy (different from average of overall accuracies)
    average_per_row_accuracy = np.mean(row_accuracies)
    print(
        f"\nAverage Per-Row Accuracy (Avg of {num_samples} row accuracies): {average_per_row_accuracy:.4f}"
    )


# --- Script Entry Point ---
if __name__ == "__main__":
    # Example Usage:
    # python your_script_name.py path/to/your/best_model.keras path/to/your/test_data.csv

    if len(sys.argv) != 3:
        print("Usage: python script_name.py <path_to_model.keras> <path_to_data.csv>")
        sys.exit(1)

    model_file = sys.argv[1]
    data_file = sys.argv[2]

    evaluate_csv_per_row(model_file, data_file)
