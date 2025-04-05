#!/usr/bin/env python3

import sys
import os
import chess.pgn
import random
import csv
import argparse
from bitboard import create_bitboards, print_bitboard

def read_pgn(pgn_fp):
    """
    Read a game from a PGN file.
    
    Args:
        pgn_fp: File pointer to the PGN file
        
    Returns:
        chess.pgn.Game or None: The game object if successful, None if no more games or error
    """
    try:
        game = chess.pgn.read_game(pgn_fp)
        return game
    except Exception as e:
        print(f"Error reading PGN file: {e}")
        return None

def write_data(output_fp, fen, label):
    """
    Write a row to the CSV file containing the FEN and label.
    
    Args:
        output_fp: File pointer to the output CSV file
        fen: The FEN string representing the board position
        label: The label for this position (-1, 0, or 1)
    """
    writer = csv.writer(output_fp, quoting=csv.QUOTE_ALL)
    writer.writerow([fen, label])

def main():
    parser = argparse.ArgumentParser(description='Generate chess position data from PGN files.')
    parser.add_argument('pgn_path', help='Path to the input PGN file')
    parser.add_argument('output_path', help='Path to the output CSV file')
    parser.add_argument('-c', '--count', type=int, default=100000, 
                       help='Number of games to process (default: process all games)')
    parser.add_argument('-o', '--overwrite', action='store_true',
                       help='Overwrite output file if it exists')
    
    args = parser.parse_args()

    if not os.path.exists(args.pgn_path):
        print(f"Error: The file {args.pgn_path} does not exist.")
        sys.exit(1)

    if os.path.exists(args.output_path) and not args.overwrite:
        print(f"Error: The file {args.output_path} already exists. Use -o to overwrite.")
        sys.exit(1)

    mode = 'w' if args.overwrite else 'x'

    with open(args.pgn_path, 'r') as pgn_fp, open(args.output_path, mode) as output_fp:
        # Write header row
        writer = csv.writer(output_fp, quoting=csv.QUOTE_ALL)
        writer.writerow(['fen', 'label'])

        games_processed = 0
        for _ in range(args.count):
            game = read_pgn(pgn_fp)
            if game is None:
                print(f"\nReached end of PGN file after processing {games_processed} games.")
                break

            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
            fen = board.fen()

            # TODO: create bitboards and analyze to create a label
            # bitboards = create_bitboards(board)
            # create a random label for now        
            label = random.choice([-1, 0, 1])

            write_data(output_fp, fen, label)
            games_processed += 1

if __name__ == "__main__":
    main()
