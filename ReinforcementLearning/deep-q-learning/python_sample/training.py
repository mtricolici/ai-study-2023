
import os
import sys
from ai import deep_qlearn
current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(current, "../../../../pylibs/snake")))
from snake import SnakeGame

def train():
    print("Training started...")
    game = SnakeGame(10, True)
    game.max_moves_without_apple = 30
    ai = deep_qlearn.DeepQLearning(game)
    ai.train_multiple_games(1000)
    print("Training finished!")

def train_continue():
    print("Continuing training...")

