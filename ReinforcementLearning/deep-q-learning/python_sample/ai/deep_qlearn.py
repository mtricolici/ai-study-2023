from os import path
import random
import numpy as np
import tensorflow as tf
import sys
import os
import gc
from ai.stats import Statistics

current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(current, "../../../../pylibs/snake")))
from snake import SnakeGame

OUTPUT_SIZE=3 # Turn Left, Keep Forward, Turn Right
HIDDEN1_SIZE=60 # Number of neurons in hidden layer 1
#HIDDEN2_SIZE=30 # Number of neurons in hidden layer 2

class DeepQLearning:
    def __init__(self, game:SnakeGame):
        self.game = game
        self.state_size = len(game.get_state_for_nn())
        self.learning_rate = 0.01
        self.discount_factor = 0.99
        self.initial_epsilon = 1.0
        self.final_epsilon = -0.5
        self.batch_size=64
        self.max_memory = 1000
        self.memory = []
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(HIDDEN1_SIZE, input_dim=self.state_size, activation='sigmoid'),
#            tf.keras.layers.Dense(HIDDEN2_SIZE, activation='relu'),
            tf.keras.layers.Dense(OUTPUT_SIZE, activation='sigmoid')
        ])

        #model_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model_optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)

        model.compile(loss='binary_crossentropy',
                      optimizer=model_optimizer,
                      metrics=['accuracy'])
        return model

    def _remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.max_memory:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))
    
    def _act(self, state, epsilon:float):
        if np.random.rand() <= epsilon:
            return random.randrange(OUTPUT_SIZE)
        else:
            return np.argmax(self._predict(state))

    def _predict(self, state):
        return self.model.predict(np.array([state]), verbose=0)[0]

    def demo_predict_next_direction(self):
        if self.game.game_over:
            return
        state = self.game.get_state_for_nn()
        action = np.argmax(self._predict(state))
        self._change_direction(action)

    def _change_direction(self, action:int):
        if action == 0:
            self.game.turn_left()
        elif action == 1:
            self.game.turn_right()

    def _replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        memory_states = []
        for state, action, reward, next_state, done in minibatch:
            memory_states.append(state) # index 0
            memory_states.append(next_state) # index 1

        predicted = self.model.predict(memory_states, batch_size=self.batch_size*2, verbose=0)
        memory_states = None
        _ = gc.collect()

        states, targets = [], []
        predicted_index = 0
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_actions = predicted[predicted_index+1] #self._predict(next_state)
                target = reward + self.discount_factor * np.amax(next_actions)
            target_f = predicted[predicted_index] #self._predict(state)
            target_f[action] = target
            states.append(state)
            targets.append(target_f)
            predicted_index+=1

        self.model.fit(
            np.array(states),
            np.array(targets),
            batch_size = self.batch_size,
            epochs=1, verbose=0)
        states = None
        targets = None
        _ = gc.collect()

    def train_multiple_games(self, num_epochs:int, percent_interval:int):
            print(f"Deep Q-Learning started for {num_epochs} games ...")

            stats = Statistics(self.game)
            interval_epochs = num_epochs * percent_interval // 100

            epsilonDecrement = (self.initial_epsilon - self.final_epsilon) / num_epochs
            epsilon = self.initial_epsilon

            for epoch in range(num_epochs):
                self._play_random_game(epsilon, stats)
                stats.print_progress(epoch, num_epochs, interval_epochs, epsilon)
                # Decrement epsilon over time
                epsilon -= epsilonDecrement
                if stats.get_memory_usage_mb() >= 8192:
                    print("Stop the training!!! Consuming more than 8 GB of ram")
                    break

    def _play_random_game(self, epsilon:float, stats:Statistics):
        self.game.reset()
        state = self.game.get_state_for_nn()
        done = False

        while not done:
            action = self._act(state, epsilon)
            self._change_direction(action)
            self.game.next_tick()

            next_state = self.game.get_state_for_nn()
            reward = self.game.reward
            done = self.game.game_over
            self._remember(state, action, reward, next_state, done)
            state = next_state
            stats.collect() # Collect statistics

        self._replay()

    def save(self, file_name:str):
        self.model.save_weights(file_name)

    def load(self, file_name:str):
         self.model.load_weights(file_name)
