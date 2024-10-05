import os
import chess
def save_pgn(pgn_moves, white_name, black_name):
    pgn = " ".join(pgn_moves)
    file_path = f'./pgns/{white_name}_vs_{black_name}.pgn'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as file:
        file.write(pgn)
    
    print(f"PGN saved at {file_path}")


def print_top_moves(move_values,s):
    move_values = sorted(move_values, key=lambda x: x[0], reverse=s.board.turn)[:5]
    print("Top moves:")
    for i, (value, move) in enumerate(move_values[0:3]):
        move_notation = s.board.san(move)
        print(f"  Move {i + 1}: {move_notation} with value {value}")

def explore_leaves(s, v):
    ret = []
    start = time.time()
    v.reset()
    bval = v(s)
    cval, ret = bot_minimax(s, v, 0, a=-MAXVAL, b=MAXVAL, big=True)
    eta = time.time() - start
    print("%.2f -> %.2f: explored %d nodes in %.3f seconds" % (bval, cval, v.count, eta))
    return ret
