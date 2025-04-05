import chess

def create_bitboards_pieces(board):
    bitboards = {}
    
    for piece_type in chess.PIECE_TYPES:
        # Create bitboard for white pieces
        bitboards[f'w{chess.piece_symbol(piece_type)}'] = board.pieces(piece_type, chess.WHITE)
        # Create bitboard for black pieces
        bitboards[f'b{chess.piece_symbol(piece_type)}'] = board.pieces(piece_type, chess.BLACK)
    
    return bitboards

def create_bitboards(game):
    # Get the board position from the game
    board = game.board()
    bitboards = {}    
    bitboards.update(create_bitboards_pieces(board))
    return bitboards

def print_bitboard(key, bb):
    print(f"{key}:\n{bb}\n") 