import os
import chess
import time
import chess.pgn
import torch
from state import State
from train import Net

MAXVAL = 1e6
class ClassicValuator(object):
    values = {chess.PAWN: 1,
              chess.KNIGHT: 3,
              chess.BISHOP: 3,
              chess.ROOK: 5,
              chess.QUEEN: 9,
              chess.KING: 0}

    def __init__(self):
        self.name = "minimax"
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



def bot_move(s, v):
    # bot move
    ret = []
    v.reset()
    bval = v(s)
    cval, ret = bot_minimax(s, v, 0, a=-MAXVAL, b=MAXVAL, big=True)
    best_move = sorted(ret, key=lambda x: x[0], reverse=s.board.turn)[0][1]
    move_notation = s.board.san(best_move)
    s.board.push(best_move)
  
    return move_notation
