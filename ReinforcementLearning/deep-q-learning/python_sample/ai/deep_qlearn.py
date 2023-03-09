from os import path
import random
import time
import numpy as np
import tensorflow as tf
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(current, "../../../../pylibs/snake")))
from snake import SnakeGame

OUTPUT_SIZE=3 # Turn Left, Keep Forward, Turn Right
HIDDEN1_SIZE=60 # Number of neurons in hidden layer 1
HIDDEN2_SIZE=30 # Number of neurons in hidden layer 2

class DeepQLearning:
    def __init__(self, game:SnakeGame):
        self.game = game
        self.state_size = len(game.get_state_for_nn())
        self.action_size = OUTPUT_SIZE
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size=32
        self.memory = []
        self.model = self._build_model()
        
    def _build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(HIDDEN1_SIZE, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(HIDDEN2_SIZE, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        #model.compile(loss='mse', optimizer='adam')
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def _remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def _act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self._predict(state))

    def _predict(self, state):
        return self.model.predict(np.array([state]), verbose=0)[0]

    def _replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_actions = self._predict(next_state)
                target = reward + self.discount_factor * np.amax(next_actions)
            target_f = self._predict(state)
            target_f[action] = target
            states.append(state)
            targets.append(target_f)

        self._train(states, targets)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _train(self, states, targets):
        start_time = time.time()
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        print(f"---_train took : {time.time() - start_time:.4f} seconds")

    def train_multiple_games(self, num_epochs):
            print(f"Deep Q-Learning started for {num_epochs} games ...")
            percent_interval = 10
            interval_epochs = num_epochs * percent_interval // 100

            for epoch in range(num_epochs):
                print(f"==== game {epoch} of {num_epochs}")
                self.game.reset()
                state = self.game.get_state_for_nn()
                done = False
                
                start_time = time.time()
                while not done:
                    action = self._act(state)
                    if action == 0:
                        self.game.turn_left()
                    elif action == 1:
                        self.game.turn_right()
                    self.game.next_tick()

                    next_state = self.game.get_state_for_nn()
                    reward = self.game.reward
                    done = self.game.game_over
                    self._remember(state, action, reward, next_state, done)
                    state = next_state
                print(f"---game end : {time.time() - start_time:.4f} seconds")
                self._replay()
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

                if epoch % interval_epochs == 0:
                    percent_complete = (epoch / num_epochs) * 100
                    print(f"Processing {percent_complete:.0f}% complete")
                
                #self.memory = []

    def save(self, file_name:str):
        self.model.save_weights(file_name)

    def load(self, file_name:str):
         self.model.load_weights(file_name)

