


import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(current, "../../../../pylibs/snake")))
from snake import SnakeGame

class Statistics:
    def __init__(self, game:SnakeGame):
        self.game = game
        self.max_reward = 0.0
        self.sum_rewards = 0.0
        self.max_moves = 0
        self.sum_moves = 0
        self.max_apples = 0
        self.sum_apples = 0
        self.games = 0

    def collect(self):
        if self.game.game_over:
            self.games += 1

        if self.game.reward > self.max_reward:
            self.max_reward = self.game.reward
        if self.max_moves < self.game.moves_made:
            self.max_moves = self.game.moves_made
        if self.max_apples < self.game.consumed_apples:
            self.max_apples = self.game.consumed_apples

        self.sum_rewards += self.game.reward
        self.sum_moves += self.game.moves_made
        self.sum_apples += self.game.consumed_apples

    def print_progress(self, epoch:int, num_epochs:int, interval_epochs:int, epsilon:float):
        if epoch > 0 and epoch % interval_epochs == 0:
            percent_complete = (epoch / num_epochs) * 100
            print(f"==> Processing {percent_complete:.0f}% complete. {epoch} of {num_epochs} games.")
            self._print(epsilon)

    def _print(self, epsilon:float):
        avgReward = self.sum_rewards / self.games
        avgMoves = self.sum_moves / self.games
        avgApples = self.sum_apples / self.games

        exp = epsilon*100
        if exp < 0: # It may be less than 0 - depends of epsilon_decay
            exp = 0.0

        print(f"->apples(max:{self.max_apples}, avg:{avgApples:.3f})"
            f" moves(max:{self.max_moves}, avg:{avgMoves:.3f})"
            f" reward(max:{self.max_reward}, avg:{avgReward:.3f}). Exploration: {exp:.0f}%")
        self.games = 0
        self.max_reward = 0.0
        self.sum_rewards = 0.0
        self.max_moves = 0
        self.sum_moves = 0
        self.max_apples = 0
        self.sum_apples = 0