#!/usr/bin/env python3

import sys
import os
import chess.pgn
from bitboard import create_bitboards, print_bitboard

def read_pgn(pgn_fp):
    game = chess.pgn.read_game(pgn_fp)
    return game

def write_data(output_fp, bitboards, overwrite):
    # Stub: Replace with actual implementation
    pass

def main():
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: script.py <count> <pgn_path> <output_path> [-o]")
        sys.exit(1)

    count = int(sys.argv[1])
    pgn_path = sys.argv[2]
    output_path = sys.argv[3]
    overwrite = '-o' in sys.argv

    if not os.path.exists(pgn_path):
        print(f"Error: The file {pgn_path} does not exist.")
        sys.exit(1)

    if os.path.exists(output_path) and not overwrite:
        print(f"Error: The file {output_path} already exists. Use -o to overwrite.")
        sys.exit(1)

    mode = 'w' if overwrite else 'x'

    with open(pgn_path, 'r') as pgn_fp, open(output_path, mode) as output_fp:
        for _ in range(count):
            game = read_pgn(pgn_fp)
            bitboards = create_bitboards(game)
            for key, bb in bitboards.items():
                print_bitboard(key, bb)
            write_data(output_fp, bitboards, overwrite)

if __name__ == "__main__":
    main()
