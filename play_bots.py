import os
import chess
import time
import chess.pgn
import torch
from state import State
from train import Net
from leela import Leela_Network
from mini_max import bot_move, ClassicValuator
from helper import save_pgn

arch = [Net, Leela_Network]

class White(object):
    def __init__(self,name,arch_indx=0):
        # Load model and weights
        self.model_name = name
        self.arch_indx = arch_indx
        self.architecture = arch[self.arch_indx]
        self.model = self.architecture()
        vals = torch.load(f"nets/{name}.pth", map_location=lambda storage, loc: storage)
        self.model.load_state_dict(vals, strict=False)

    def __call__(self, s):
        # Serialize the board state for input to the model
        brd = s.serialize()[None]
        output = self.model(torch.tensor(brd).float())
        if self.arch_indx == 0:
            return float(output.data[0][0])
        return float(output.item())


class Black(object):
    def __init__(self,name,arch_indx=0):
        self.model_name = name
        self.arch_indx = arch_indx
        self.architecture = arch[self.arch_indx]
        self.model = self.architecture()
        vals = torch.load(f"nets/{name}.pth", map_location=lambda storage, loc: storage)
        self.model.load_state_dict(vals)

    def __call__(self, s):
        brd = s.serialize()[None]
        output = self.model(torch.tensor(brd).float())
        if self.arch_indx == 0:
            return float(output.data[0][0])
        return float(output.item())


def white_move(s, v):
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
        # print(f"  Move {i + 1}: {move_notation} with value {value}")
    
    best_move = move_values[0][1]  # Select the best move
    move_notation = s.board.san(best_move)
    s.board.push(best_move)
  
    return move_notation

def black_move(s, v):
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

def play(white,black):
    s = State()
    print(f"Starting {white.model_name} v/s {black.model_name}...")
    pgn_moves = []
    gn = 1  # Game number (move number)

    while not s.board.is_game_over():
        white_played = white_move(s, white)
        
        if white_played: 
            if gn > 1:  
                pgn_moves[-1] += " " 
            pgn_moves.append(f"{gn}. {white_played}")
            
            # print(str(s.board))
            
            if s.board.is_game_over():
                break
        
            black_played = black_move(s, black)
            if black_played:
                pgn_moves[-1] += f" {black_played}"  
            
            gn += 1  # Increment game number

    print("Game Over!")
    print("Result:", s.board.result())

    save_pgn(pgn_moves, white_name, black_name)

def classical_play(color, test=True):
    s = State() 
    print("Starting self-play...")

    pgn_moves = []
    gn = 1

    if test:
        black = ClassicValuator()  # AI black
        white_name = color.model_name
        black_name = "minimax"
    else:
        white = ClassicValuator()  # AI white
        white_name = "minimax"
        black_name = color.model_name
    
    print(f"Starting {white_name} v/s {black_name}...")
    
    pgn_moves = []
    gn = 1


    while not s.board.is_game_over():
        if test:  # White is playing against ClassicValuator as black
            white_played = white_move(s, color)
            if white_played:
                if gn > 1:
                    pgn_moves[-1] += " "
                pgn_moves.append(f"{gn}. {white_played}")
                
                if s.board.is_game_over():
                    break
                
                black_played = bot_move(s, black)
                if black_played:
                    pgn_moves[-1] += f" {black_played}"
                
                gn += 1 

        else:  # ClassicValuator as white is playing against black
            white_played = bot_move(s, white)
            if white_played:
                if gn > 1:
                    pgn_moves[-1] += " "
                pgn_moves.append(f"{gn}. {white_played}")
                
                if s.board.is_game_over():
                    break
                
                black_played = black_move(s, color)
                if black_played:
                    pgn_moves[-1] += f" {black_played}"
                
                gn += 1 

    print("Game Over!")
    print("Result:", s.board.result())

    # Save PGN file
    save_pgn(pgn_moves, white_name, black_name)



if __name__ == "__main__":

    # Bot : Name, Arch_index
    white = White("nakamura",0)
    black = Black("ding",0)
    # play(white,black)

    # Classical
    
    classical_play(black,False)
