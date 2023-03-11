
import os
import sys
from ai import deep_qlearn
current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(current, "../../../../pylibs/snake")))
from snake import SnakeGame

def train(size:int, iterations:int, percent_interval:int, saveFileName:str):
    print("Training started...")
    game = SnakeGame(size, True)
    game.max_moves_without_apple = size * 3

    ai = deep_qlearn.DeepQLearning(game)
    ai.train_multiple_games(iterations, percent_interval)
    print("Training finished!")

    ai.save(saveFileName)
    print(f"Weights saved to '{saveFileName}'")

def train_continue():
    print("Continuing training...")

