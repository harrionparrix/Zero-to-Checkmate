# zero to checkmate: RL Based Chess Engine

## Overview

**ChessBot** is a chess-playing bot that utilizes reinforcement learning and Monte Carlo methods to make strategic decisions during gameplay. The bot is designed to simulate self-play, evaluate positions, and make intelligent moves based on a trained neural network model.

## Features

- **Minimax Algorithm**: Implements a minimax algorithm with alpha-beta pruning for optimal move selection.
- **Value Estimation**: Utilizes a neural network for board state evaluation, alongside classic heuristic evaluations.
- **Self-Play**: Capable of playing against itself for training and testing purposes.
- **PGN Export**: Saves the game in PGN format for analysis and record-keeping.

## Installation

### Prerequisites

1. **Python 3.x**: Ensure you have Python 3 installed on your machine.
2. **PyTorch**: Install PyTorch following the instructions from the [official website](https://pytorch.org/get-started/locally/).
3. **chess**: Install the `python-chess` library.
   ```bash
   pip install python-chess
   ```

## Architecture

-**train.py**: Implements a CNN architecture for self-play.

-**leela.py** Implements the LEELA architecture (https://lczero.org/dev/backend/nn/)
