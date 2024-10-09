import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import chess
import chess.pgn
import chess.engine
import random
from math import log, sqrt, e, inf
from helper import save_pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class AlphaZero(nn.Module):
    def __init__(self, num_residual_blocks=19):
        super(AlphaZero, self).__init__()
        self.conv_input = nn.Conv2d(5, 256, kernel_size=3, padding=1) 
        
        # Stack of Residual Blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residual_blocks)]
        )
        
        self.policy_head = nn.Conv2d(256, 2, kernel_size=1)  # Policy head
        self.value_head = nn.Conv2d(256, 1, kernel_size=1)    # Value head
        
        self.fc_policy = nn.Linear(2 * 8 * 8, 64)  # Linear layer for policy
        self.fc_value = nn.Linear(8 * 8, 1)         # Linear layer for value

    def forward(self, x):
        x = F.relu(self.conv_input(x))  # Input convolution
        x = self.residual_blocks(x)      # Residual blocks
        
        policy_logits = self.policy_head(x)  # (B, 2, 8, 8)
        policy = F.softmax(policy_logits.view(-1, 2 * 8 * 8), dim=1)  # (B, 2*64)

        # Value head
        value = self.value_head(x)           # (B, 1, 8, 8)
        value = F.tanh(value.view(-1, 8 * 8))  # (B, 64)

        return policy, value

def board_to_tensor(board):
    tensor = torch.zeros((5, 8, 8), dtype=torch.float32)
    piece_map = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
    }
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            row, col = divmod(i, 8)
            tensor[0 if piece.color == chess.WHITE else 1, row, col] = piece_map[piece.symbol()]
    return tensor.unsqueeze(0)  # Add batch dimension


import chess.engine

def evaluate_model(model, stockfish_path, num_games=100):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    wins, draws, losses = 0, 0, 0

    for _ in range(num_games):
        board = chess.Board()
        
        while not board.is_game_over():
            board_tensor = board_to_tensor(board)
            policy, _ = model(board_tensor)

            legal_moves = list(board.legal_moves)
            move_probs = policy[0].detach().numpy()
            move_probs_full = np.zeros(len(legal_moves))
            move_indices = {move: i for i, move in enumerate(legal_moves)}

            for move in legal_moves:
                move_index = move_indices[move]
                move_probs_full[move_index] = move_probs[move_index]

            if np.sum(move_probs_full) > 0:
                move_probs_full /= np.sum(move_probs_full)
            else:
                move_probs_full = np.ones(len(legal_moves)) / len(legal_moves)

            selected_move = np.random.choice(legal_moves, p=move_probs_full)
            board.push(selected_move)

            if board.is_game_over():
                break
            
            stockfish_move = engine.play(board, chess.engine.Limit(time=2.0)).move
            board.push(stockfish_move)

        result = board.result()
        if result == '1-0':
            wins += 1
        elif result == '0-1':
            losses += 1
        else:
            draws += 1

    engine.quit()
    return wins, draws, losses


def self_play_with_stockfish(model, num_games=1000):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    engine_path = r'./stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe'
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    
    for game in range(num_games):
        board = chess.Board()
        states, rewards = [], []

        while not board.is_game_over():
            board_tensor = board_to_tensor(board)
            policy, value = model(board_tensor)

            legal_moves = list(board.legal_moves)
            move_probs = policy[0].detach().numpy()

            move_probs_full = np.zeros(len(legal_moves))
            move_indices = {move: i for i, move in enumerate(legal_moves)}
            for move in legal_moves:
                move_index = move_indices[move]
                move_probs_full[move_index] = move_probs[move_index]

            if np.sum(move_probs_full) > 0:
                move_probs_full /= np.sum(move_probs_full)
            else:
                move_probs_full = np.ones(len(legal_moves)) / len(legal_moves)

            selected_move = np.random.choice(legal_moves, p=move_probs_full)
            states.append(board_tensor)
            rewards.append(0)
            board.push(selected_move)

            if board.is_game_over():
                break
            stockfish_move = engine.play(board, chess.engine.Limit(time=2.0)).move
            board.push(stockfish_move)

        if board.result() == '1-0':
            reward = 1
        elif board.result() == '0-1':
            reward = -1
        else:
            reward = 0

        # Update rewards for all states visited in the game
        for state in states:
            state_tensor = state.view(1, 5, 8, 8)
            _, value = model(state_tensor)
            value_flat = value.view(-1)
            loss = F.mse_loss(value_flat, torch.tensor([reward], dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Game {game + 1}/{num_games} complete.")

        # Evaluatin
        if (game + 1) % 100 == 0:
            wins, draws, losses = evaluate_model(model, engine_path, num_games=1)
            print(f"Evaluation after {game + 1} games: Wins: {wins}, Draws: {draws}, Losses: {losses}")

    engine.quit()

if __name__ == "__main__":
    model = AlphaZero()
    self_play_with_stockfish(model, num_games=1000)
    torch.save(model.state_dict(), "nets/alpha_stockfish.pth")

