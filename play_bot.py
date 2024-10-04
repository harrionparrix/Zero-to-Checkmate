from __future__ import print_function
import os
import chess
import time
import chess.svg
import traceback
import base64
from state import State
import chess.pgn
class Valuator(object):
    def __init__(self):
        import torch
        from train import Net
        vals = torch.load("nets/ding.pth", map_location=lambda storage, loc: storage)
        self.model = Net()
        self.model.load_state_dict(vals)

    def __call__(self, s):
        brd = s.serialize()[None]
        output = self.model(torch.tensor(brd).float())
        return float(output.data[0][0])

MAXVAL = 1e6
class ClassicValuator(object):
    values = {chess.PAWN: 1,
              chess.KNIGHT: 3,
              chess.BISHOP: 3,
              chess.ROOK: 5,
              chess.QUEEN: 9,
              chess.KING: 0}

    def __init__(self):
        self.reset()
        self.memo = {}

    def reset(self):
        self.count = 0

    def __call__(self, s):
        self.count += 1
        key = s.key()
        if key not in self.memo:
            self.memo[key] = self.value(s)
        return self.memo[key]

    def value(self, s):
        b = s.board
        if b.is_game_over():
            if b.result() == "1-0":
                return MAXVAL
            elif b.result() == "0-1":
                return -MAXVAL
            else:
                return 0

        val = 0.0
        pm = s.board.piece_map()
        for x in pm:
            tval = self.values[pm[x].piece_type]
            if pm[x].color == chess.WHITE:
                val += tval
            else:
                val -= tval
        bak = b.turn
        b.turn = chess.WHITE
        val += 0.1 * b.legal_moves.count()
        b.turn = chess.BLACK
        val -= 0.1 * b.legal_moves.count()
        b.turn = bak

        return val

def bot_minimax(s, v, depth, a, b, big=False):
    if depth >= 3 or s.board.is_game_over():
        return v(s)
    if depth >= 5 or s.board.is_game_over():
        return v(s)
    
    turn = s.board.turn
    if turn == chess.WHITE:
        ret = -MAXVAL
    else:
        ret = MAXVAL
    if big:
        bret = []

    isort = []
    for e in s.board.legal_moves:
        s.board.push(e)
        isort.append((v(s), e))
        s.board.pop()
    move = sorted(isort, key=lambda x: x[0], reverse=s.board.turn)

    if depth >= 3:
        move = move[:10]

    for e in [x[1] for x in move]:
        s.board.push(e)
        tval = bot_minimax(s, v, depth+1, a, b)
        s.board.pop()
        if big:
            bret.append((tval, e))
        if turn == chess.WHITE:
            ret = max(ret, tval)
            a = max(a, ret)
            if a >= b:
                break  # b cut-off
        else:
            ret = min(ret, tval)
            b = min(b, ret)
            if a >= b:
                break  # a cut-off
    if big:
        return ret, bret
    else:
        return ret

def explore_leaves(s, v):
    ret = []
    start = time.time()
    v.reset()
    bval = v(s)
    cval, ret = bot_minimax(s, v, 0, a=-MAXVAL, b=MAXVAL, big=True)
    eta = time.time() - start
    print("%.2f -> %.2f: explored %d nodes in %.3f seconds %d/sec" % (bval, cval, v.count, eta, int(v.count/eta)))
    return ret

# Instantiate the evaluator
v = ClassicValuator()

def bot_move(s, v):
    # bot move
    move = sorted(explore_leaves(s, v), key=lambda x: x[0], reverse=s.board.turn)[:5]  # Limit to top 5 moves
    if len(move) == 0:
        return None  # No valid moves, return None
    
    print("Top moves:")
    for i, m in enumerate(move[0:3]):
        move_notation = s.board.san(m[1])
        print(f"  Move {i + 1}: {move_notation} with value {m[0]}")
    
    best_move = move[0][1]
    move_notation = s.board.san(best_move)
    print("white" if s.board.turn else "black", "moving", move_notation)
    s.board.push(best_move)
  
    return move_notation

def self_play():
    s = State() 
    print("Starting self-play...")


    pgn_moves = []
    gn = 1

    while not s.board.is_game_over() and gn<4:
        white_move = bot_move(s, v)
        
        if white_move: 
            if gn > 1:  
                pgn_moves[-1] += " "
            pgn_moves.append(f"{gn}. {white_move}")
            
            print(str(s.board))
            
            if s.board.is_game_over():
                break
        
            black_move = bot_move(s, v)
            if black_move:
                pgn_moves[-1] += f"{black_move}"  
            
            gn += 1 

    print("Game Over!")
    print("Result:", s.board.result())

    pgn = "".join(pgn_moves)

    file_path = './game.pgn'
    with open(file_path, 'w') as file:
        file.write(pgn)

if __name__ == "__main__":
    self_play()