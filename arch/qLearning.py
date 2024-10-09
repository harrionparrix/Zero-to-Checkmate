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

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state  # The board configuration (chess.Board object)
        self.parent = parent  # Parent node in the tree
        self.children = []  # List of child nodes
        self.n = 0  # Number of visits to this node
        self.v = 0  # Total reward accumulated at this node
        self.move = move  # The move that led to this node
        self.Q = {}  # Q-values for each move (action) from this state

    def add_child(self, child):
        self.children.append(child)

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.state.legal_moves))


def ucb1(curr_node, exploration_param=2, e = 1e-6):
    """UCB1 formula for balancing exploration and exploitation."""
    if curr_node.n == 0:
        return inf  # Prioritize unvisited nodes
    return (curr_node.v / curr_node.n) + exploration_param * sqrt(log(curr_node.parent.n + e) / curr_node.n)


def rollout(curr_node):
    """Simulate a random playthrough to the end of the game."""
    board = curr_node.state
    while not board.is_game_over():
        legal_moves = list(board.legal_moves)
        move = random.choice(legal_moves)
        board.push(move)

    # Return game result: +1 for white win, -1 for black win, 0.5 for draw
    result = board.result()
    if result == '1-0':
        return 1
    elif result == '0-1':
        return -1
    return 0.5


def expand(curr_node):
    """Expand a node by adding one unexplored child."""
    legal_moves = list(curr_node.state.legal_moves)
    unexplored_moves = [move for move in legal_moves if move not in [child.move for child in curr_node.children]]
    if unexplored_moves:
        move = random.choice(unexplored_moves)
        tmp_state = curr_node.state.copy()
        tmp_state.push(move)
        child = Node(state=tmp_state, parent=curr_node, move=move)
        curr_node.add_child(child)
        return child
    return random.choice(curr_node.children)


def best_child(node, white, epsilon=0.1):
    """Select the best child node based on Q-values using epsilon-greedy strategy."""
    if random.uniform(0, 1) < epsilon:
        # Explore: Pick a random move
        return random.choice(node.children)
    
    # Exploit: Choose the child node with the highest Q-value
    return max(node.children, key=lambda child: node.Q.get(child.move, 0))


def backup(node, reward, learning_rate=0.1, discount_factor=0.9):
    """Backpropagate the reward up the tree and update Q-values."""
    while node:
        # Get the move that led to this node
        move = node.move
        
        if node.parent is not None:
            # Get current Q-value for the move that led to this node
            current_q = node.parent.Q.get(move, 0)  # Default Q-value is 0 if it hasn't been visited

            # Q-learning update rule
            future_q = max(node.Q.values(), default=0)  # Max Q-value for future moves
            updated_q = current_q + learning_rate * (reward + discount_factor * future_q - current_q)

            # Update the Q-value for this move in the parent node
            node.parent.Q[move] = updated_q

        # Move up the tree
        node = node.parent




def mcts_search(root, white, iterations=1000):
    """Perform MCTS search and return the best move."""
    for _ in range(iterations):
        node = root

        # UCB1 selection
        while node.is_fully_expanded() and node.children:
            node = best_child(node, white)

        # Expansion
        if not node.state.is_game_over() and not node.is_fully_expanded():
            node = expand(node)

        reward = rollout(node)

        # Backpropagation
        backup(node, reward)

    best_move = max(root.children, key=lambda child: child.n).move
    return best_move


# Chess game setup
board = chess.Board()

engine_path = r'./stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe'
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

white = True
pgn_moves = []
game = chess.pgn.Game()
iterations = 100
gn = 1
while not board.is_game_over():
    root = Node(state=board)
    white_played = mcts_search(root, True, iterations)

    if white:
        if gn > 1:  
            pgn_moves[-1] += " "
        pgn_moves.append(f"{gn}. {white_played}")

        if white_played in board.legal_moves:
            board.push(white_played)
        
        root = Node(state=board)
        black_played = mcts_search(root, False, iterations) 
        
        if black_played in board.legal_moves:
            board.push(black_played)
        if black_played:
            pgn_moves[-1] += f" {black_played}"  
        gn += 1
    print("Move Number: ", gn)
    

print(board)
save_pgn(pgn_moves, "QLearn", "QLearn")
print(board.result())
game.headers["Result"] = board.result()

# Close the chess engine
engine.quit()