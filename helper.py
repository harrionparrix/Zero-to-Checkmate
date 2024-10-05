import os
import chess
def save_pgn(pgn_moves, white_name, black_name):
    pgn = " ".join(pgn_moves)
    file_path = f'./pgns/{white_name}_vs_{black_name}.pgn'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as file:
        file.write(pgn)
    
    print(f"PGN saved at {file_path}")