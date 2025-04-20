#!/usr/bin/env python3

import argparse
from train import train

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a CNN model on chess data.")
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the CSV file containing FEN strings and labels",
    )
    args = parser.parse_args()  # Parse the arguments

    train(args.data_path)
