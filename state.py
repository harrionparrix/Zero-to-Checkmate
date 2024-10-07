import chess
import numpy as np


class State(object):
  def __init__(self, board=None):
    if board is None:
      self.board = chess.Board()
    else:
      self.board = board
  def key(self):
    return (self.board.board_fen(), self.board.turn, self.board.castling_rights, self.board.ep_square)

  def serialize(self):
      assert self.board.is_valid()

      # 1. Initialize board state array (5 layers of 8x8 matrix)
      state = np.zeros((5, 8, 8), dtype=np.uint16)

      # 2. Piece mapping: 1-6 for white pieces, 9-14 for black pieces
      piece_map = {
          'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,   # White
          'p': 9, 'n': 10, 'b': 11, 'r': 12, 'q': 13, 'k': 14  # Black
      }

      # 3. Encode piece positions into the 8x8 board
      for i in range(64):
          piece = self.board.piece_at(i)
          if piece is not None:
              row, col = divmod(i, 8)  # Convert 1D index to 2D (row, col)
              state[0, row, col] = piece_map[piece.symbol()]

      # 4. Encode castling rights
      # top 2 bits for white, bottom 2 bits for black
      if self.board.has_queenside_castling_rights(chess.WHITE):
          state[0, 7, 0] |= 1  # White queenside
      if self.board.has_kingside_castling_rights(chess.WHITE):
          state[0, 7, 7] |= 2  # White kingside
      if self.board.has_queenside_castling_rights(chess.BLACK):
          state[0, 0, 0] |= 1  # Black queenside
      if self.board.has_kingside_castling_rights(chess.BLACK):
          state[0, 0, 7] |= 2  # Black kingside

      # 5. Encode en passant square
      if self.board.ep_square is not None:
          row, col = divmod(self.board.ep_square, 8)
          state[1, row, col] = 1  # Mark en passant square

      # 6. Encode turn (binary flag for which player's turn it is)
      state[2, :, :] = self.board.turn  # 1 if white to move, 0 if black

      # 7. Encode halfmove clock (for the 50-move rule)
      halfmove_clock = self.board.halfmove_clock
      state[3, :, :] = halfmove_clock  # Same value across the entire layer

      # 8. Encode fullmove number (starting at 1)
      fullmove_number = self.board.fullmove_number
      state[4, :, :] = fullmove_number  # Same value across the entire layer

      return state
  def edges(self):
    return list(self.board.legal_moves)

if __name__ == "__main__":
  s = State()
  print(s.edges())

