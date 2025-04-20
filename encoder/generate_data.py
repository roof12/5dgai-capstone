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


def write_data(
    output_fp,
    fen,
    material_cmp,
    castle_white_kingside,
    castle_white_queenside,
    castle_black_kingside,
    castle_black_queenside,
):
    """
    Write a row to the CSV file containing the FEN and labels.

    Args:
        output_fp: File pointer to the output CSV file
        fen: The FEN string representing the board position
        label: The label for this position (-1, 0, or 1)
    """
    writer = csv.writer(output_fp, quoting=csv.QUOTE_ALL)
    writer.writerow(
        [
            fen,
            material_cmp,
            castle_white_kingside,
            castle_white_queenside,
            castle_black_kingside,
            castle_black_queenside,
        ]
    )


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
        "white_kingside": bool(board.castling_rights & chess.KINGSIDE),
        "white_queenside": bool(board.castling_rights & chess.QUEENSIDE),
        "black_kingside": bool(board.castling_rights & chess.kingside),
        "black_queenside": bool(board.castling_rights & chess.queenside),
    """
    parser = argparse.ArgumentParser(
        description="Generate chess position data from PGN files."
    )
    parser.add_argument("pgn_path", help="Path to the input PGN file")
    parser.add_argument("output_path", help="Path to the output CSV file")
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=9999999999,
        help="Number of games to process (default: process all games)",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists",
    )

    args = parser.parse_args()

    if not os.path.exists(args.pgn_path):
        print(f"Error: The file {args.pgn_path} does not exist.")
        sys.exit(1)

    if os.path.exists(args.output_path) and not args.overwrite:
        print(
            f"Error: The file {args.output_path} already exists. Use -o to overwrite."
        )
        sys.exit(1)

    return args


def count_material(board):
    """
    Count the material value for both white and black pieces.

    Args:
        board: chess.Board object

    Returns:
        tuple: (white_material, black_material) where each is the sum of piece values
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    white_material = 0
    black_material = 0

    # Count white pieces
    for piece_type in chess.PIECE_TYPES:
        material = len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        white_material += material

    # Count black pieces
    for piece_type in chess.PIECE_TYPES:
        material = len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
        black_material += material

    return white_material, black_material


def sign_diff(a, b):
    """
    Compare two values and return 1 if a > b, -1 if a < b, and 0 if equal.

    Args:
        a: First value to compare
        b: Second value to compare

    Returns:
        int: 1 if a > b, -1 if a < b, 0 if equal
    """
    return (a > b) - (a < b)


def main():
    args = parse_args()
    mode = "w" if args.overwrite else "x"

    with open(args.pgn_path, "r") as pgn_fp, open(args.output_path, mode) as output_fp:
        # Write header row
        writer = csv.writer(output_fp, quoting=csv.QUOTE_ALL)
        writer.writerow(
            [
                "fen",
                "material_cmp",
                "castle_white_kingside",
                "castle_white_queenside",
                "castle_black_kingside",
                "castle_black_queenside",
            ],
        )

        games_processed = 0
        for _ in range(args.count):
            game = read_pgn(pgn_fp)
            if game is None:
                print(
                    f"\nReached end of PGN file after processing {games_processed} games."
                )
                break

            board = game.board()
            mainline_moves = list(game.mainline_moves())

            # Choose a random number of moves to apply (between 0 and the number of moves in the main line)
            num_moves = random.randint(0, len(mainline_moves))

            # Apply the chosen number of moves to the board
            for move in mainline_moves[:num_moves]:
                board.push(move)

            fen = board.fen()

            # Count material for both sides
            white_material, black_material = count_material(board)
            # Compare white material with black material, returning a value in [1, 0, -1]
            material_cmp = sign_diff(white_material, black_material)

            # determine castling rights
            castle_white_kingside = int(board.has_kingside_castling_rights(chess.WHITE))
            castle_white_queenside = int(
                board.has_queenside_castling_rights(chess.WHITE)
            )
            castle_black_kingside = int(board.has_kingside_castling_rights(chess.BLACK))
            castle_black_queenside = int(
                board.has_queenside_castling_rights(chess.BLACK)
            )

            write_data(
                output_fp,
                fen,
                material_cmp,
                castle_white_kingside,
                castle_white_queenside,
                castle_black_kingside,
                castle_black_queenside,
            )
            games_processed += 1


if __name__ == "__main__":
    main()
