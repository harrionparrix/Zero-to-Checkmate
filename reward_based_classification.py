import os
import chess
import torch
import torch.nn as nn
import torch.optim as optim
from state import State
from train import Net
from alpha_zero import AlphaZero
import numpy as np

class Valuator(object):
    def __init__(self):
        self.model = Net() 
        self.model.eval() 

    def __call__(self, s):
        brd = s.serialize()[None]
        output = self.model(torch.tensor(brd).float())
        return float(output.item())

def bot_move(s, model, temperature=1.0):

    legal_moves = list(s.board.legal_moves)
    if len(legal_moves) == 0:
        return None  # No valid moves
    
    move_values = []
    for move in legal_moves:
        s.board.push(move)
        value = model(s)
        move_values.append((value, move))
        s.board.pop()
    
    values = torch.tensor([mv[0] for mv in move_values])
    if s.board.turn == chess.WHITE:
        probabilities = torch.softmax(values / temperature, dim=0)
    else:
        probabilities = torch.softmax(-values / temperature, dim=0)
    
    # Sample a move based on probabilities
    move_index = torch.multinomial(probabilities, 1).item()
    selected_move = move_values[move_index][1]
    s.board.push(selected_move)
    return selected_move

# def get_reward(s):
#     result = s.board.result()
#     if result == "1-0":  # White wins
#         return 1
#     elif result == "0-1":  # Black wins
#         return -1
#     else:
#         return 0  # Draw

import chess

def material_value(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    white_material = sum(piece_values.get(piece.piece_type, 0) for piece in board.piece_map().values() if piece.color == chess.WHITE)
    black_material = sum(piece_values.get(piece.piece_type, 0) for piece in board.piece_map().values() if piece.color == chess.BLACK)

    if board.turn == chess.WHITE:
        return white_material - black_material
    else:
        return black_material - white_material



def check_reward(board):
    if board.is_check():
        return 0.5  # small reward for check
    return 0

def self_play(model, optimizer, num_games=1000):
    for game in range(num_games):
        s = State()
        print(f"Starting game {game + 1}...")
        
        states, actions = [], []
        while not s.board.is_game_over():
            state_serialized = s.serialize()
            move = bot_move(s, model)
            if move:
                states.append(state_serialized)
                actions.append(move)
            else:
                break

        # game_reward = get_reward(s)
        game_reward = material_value(s.board)/104 + check_reward(s.board)
        print(f"Game {game + 1} result: {s.board.result()}")

        targets = torch.tensor([game_reward] * len(states), dtype=torch.float32)
        states_array = np.array(states, dtype=np.float32)
        inputs = torch.tensor(states_array, dtype=torch.float32)

        model.model.train()
        optimizer.zero_grad()
        outputs = model.model(inputs)
        outputs = outputs.view(-1)
        loss = nn.MSELoss()(outputs, targets)
        loss.backward()
        optimizer.step()
        model.model.eval()

        print(f"Game {game + 1} training loss: {loss.item()}")

        torch.save(model.model.state_dict(), f"nets/{name}.pth")

if __name__ == "__main__":
    name = "reinforce_two_basic"
    v = Valuator()
    optimizer = optim.Adam(v.model.parameters(), lr=0.01)
    self_play(v, optimizer, num_games=100)
