import os
import chess.pgn
import numpy as np
from state import State

def get_dataset(num_samples=None, name=""):
  X, Y = [], []
  gn = 0
  values = {'1/2-1/2': 0, '0-1': -1, '1-0': 1}
  
  # pgn files in the data folder
  for fn in os.listdir(f"data/{name}"):
      pgn = open(os.path.join(f"data/{name}", fn), 'r')
      while True:
          try:
              game = chess.pgn.read_game(pgn)
              if game is None:
                  break
              res = game.headers['Result']
              if res not in values:
                  continue
              value = values[res]
              board = game.board()
              for i, move in enumerate(game.mainline_moves()):
                  board.push(move)
                  ser = State(board).serialize()
                  X.append(ser)
                  Y.append(value)
              print(f"Parsing game {gn}, got {len(X)} examples")
              if num_samples is not None and len(X) > num_samples:
                  return X, Y
              gn += 1
          except Exception as e:
              print(f"Error parsing game: {e}")
              break
      pgn.close()

  X = np.array(X)
  Y = np.array(Y)

  return X, Y

if __name__ == "__main__":
    name = "Nakamura"
    X, Y = get_dataset(1_000_000, name)
    np.savez(f"processed/{name}_1M.npz", X, Y)
