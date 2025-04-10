import numpy as np
import chess

def fen_to_tensor(fen):
    """
    Convert a FEN string to a [8, 8, 12] tensor representation.
    Each channel represents a different piece type and color.
    
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


