# zero to checkmate: RL Based Chess Engine

## Overview

**zero to checkmate** is a chess-playing repository that leverages reinforcement learning and Monte Carlo methods to make strategic decisions. The repository provides various features that simulate self-play, evaluate positions, and make intelligent moves based on a neural network model trained through various techniques, including AlphaZero with Stockfish reinforcement, basic AlphaZero, Leela architecture, Monte Carlo Tree Search (MCTS), Q-learning on MCTS, and a CNN architecture for self-play.

# Resources

- Chess Games from https://www.pgnmentor.com/files.html#players
- Stockfish 17 Used (https://stockfishchess.org/download/)

## Features

- **Reinforcement Learning** - Implements Reinforcement Algorithms like QLearning and Self Learning using Stockfish
- **Minimax Algorithm**: Implements a minimax algorithm with alpha-beta pruning for optimal move selection.
- **Value Estimation**: Utilizes a neural network for board state evaluation, alongside classic heuristic evaluations.
- **Self-Play**: Capable of playing against itself for training and testing purposes.
- **PGN Export**: Saves the game in PGN format for analysis and record-keeping.

## Installation

1. **Python 3.x**: Ensure you have Python 3 installed on your machine.
2. **PyTorch**: Install PyTorch following the instructions from the [official website](https://pytorch.org/get-started/locally/).
3. **python-chess**: Install the `python-chess` library.

   ```bash
   pip install python-chess
   ```

## Architectures

1. **alpha_stockfish**: Implements RL based Alpha Zero architecture with self learning using Stockfish (https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/)

2. **MCTS**: Monte Carlo Tree Search (https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)

3. **leela.py**: Implements the LEELA architecture (https://lczero.org/dev/backend/nn/)

4. **alpha_zero.py**: Implements the base AlphaZero architecture (https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/)

5. **Q-Learning based MCTS**: Added Q Learning to improve MCTS (https://en.wikipedia.org/wiki/Q-learning)

6. **CNN.py**: Implements a CNN architecture for self-play. (https://geohot.com/)
