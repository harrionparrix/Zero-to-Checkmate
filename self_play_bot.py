import os
import chess
import time
import chess.pgn
import torch
from state import State
from train import Net
from leela import Leela_Network
class Valuator(object):
    def __init__(self):
        # Load model and weights
        vals = torch.load(f"nets/Gukesh_leela.pth", map_location=lambda storage, loc: storage)
        self.model = Leela_Network()
        print("Model state dict keys:", self.model.state_dict().keys())
        print("Loaded state dict keys:", vals.keys())
        
        # Load state dict with strict=False to ignore missing/unexpected keys
        self.model.load_state_dict(vals, strict=False)

    def __call__(self, s):
        # Serialize the board state for input to the model
        brd = s.serialize()[None]
        output = self.model(torch.tensor(brd).float())
        return float(output.item())
v = Valuator()
def bot_move(s, v):
    # Generate legal moves and evaluate them
    legal_moves = list(s.board.legal_moves)
    if len(legal_moves) == 0:
        return None  # No valid moves, return None
    
    move_values = []
    for move in legal_moves:
        s.board.push(move)
        move_value = v(s)  # Evaluate the move
        move_values.append((move_value, move))
        s.board.pop()

    # Sort moves based on evaluation value
    move_values = sorted(move_values, key=lambda x: x[0], reverse=s.board.turn)[:5]  # Limit to top 5 moves
    
    print("Top moves:")
    for i, (value, move) in enumerate(move_values[0:3]):
        move_notation = s.board.san(move)
        print(f"  Move {i + 1}: {move_notation} with value {value}")
    
    best_move = move_values[0][1]  # Select the best move
    move_notation = s.board.san(best_move)
    print("white" if s.board.turn else "black", "moving", move_notation)
    s.board.push(best_move)
  
    return move_notation

def self_play(name):
    s = State() 
    print("Starting self-play...")

    pgn_moves = []
    gn = 1  # Game number (move number)

    while not s.board.is_game_over():
        white_move = bot_move(s, v)
        
        if white_move: 
            if gn > 1:  
                pgn_moves[-1] += " "  # Format the PGN properly
            pgn_moves.append(f"{gn}. {white_move}")
            
            print(str(s.board))
            
            if s.board.is_game_over():
                break
        
            black_move = bot_move(s, v)
            if black_move:
                pgn_moves[-1] += f" {black_move}"  
            
            gn += 1  # Increment game number

    print("Game Over!")
    print("Result:", s.board.result())

    pgn = " ".join(pgn_moves)

    # Define the file path
    file_path = f'./pgns/{name}_vs_{name}.pgn'

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Write the PGN to the file, overwriting if it exists
    with open(file_path, 'w') as file:
        file.write(pgn)

if __name__ == "__main__":
    name = "Gukesh"
    # Instantiate the evaluator
    self_play(name)
