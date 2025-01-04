from random import randint
from uu import encode

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from consts import *
import numpy as np
from jacky import Game
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.optimizers import Adam

class NN():
    def __init__(self):
        # Set the learning rate and create optimizer with decay
        learning_rate = 0.001
        optimizer = Adam(learning_rate=learning_rate)

        # Create a deeper network
        model = Sequential([
            Dense(64, input_shape=(3,)),
            ReLU(),
            Dense(32),
            ReLU(),
            Dense(32),
            ReLU(),
            Dense(16),
            ReLU(),
            Dense(1)
        ])

        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        self.model = model

    def choose_action(self, game, epsilon):
        # In (1-epsilon) we will do the best action
        # In epsilon we will do a random action
        if game.sum == 0:
            return HIT

        if randint(0, 100) < epsilon * 100:
            # In epsilon we will do a random action
            return HIT if randint(0, 1) == 0 else STAY
        else:
            # In (1-epsilon) we will do the best action
            return self.best_action(game)

    def best_action(self, game):
        """
        The best action is the action that maximizes the predicted reward
        """

        my_current_reward = game.check_reward(game.my_moves[-1])
        df = encode_states(game.my_moves, my_current_reward, game.rewards)
        prediction = self.model.predict(df)
        current_state_prediction = prediction[game.sum][0]  # Get prediction for current sum
        if current_state_prediction > my_current_reward:
            return HIT
        else:
            return STAY

    def train(self, epochs=5, epsilon=0.1):
        statistics = {}
        for i in range(epochs):
            jacky_game = Game(random=True)
            while jacky_game.status == ONGOING:
                # Choose action
                action = self.choose_action(jacky_game, epsilon)
                print("Action chosen:", action)
                # make move
                jacky_game.make_move(action)

            # encode the states to df
            X = encode_states(jacky_game.my_moves, jacky_game.reward, jacky_game.rewards)
                
            # Convert DataFrame to numpy array
            y = encode_reward(X, jacky_game.reward)
            
            # train the model
            self.model.fit(X, y, epochs=1, verbose=1)

        #     # Check status
        #     if jacky_game.reward == 0:
        #         print("You lost!")
        #     else:
        #         print(f"You won {jacky_game.reward} coins!")
        #     statistics[i] = jacky_game.reward
        # print(statistics)
        # print("Average reward:", sum(statistics.values()) / epochs)
        # print("Loosed games:", len([x for x in statistics.values() if x == 0]))

    def play(self):
        game = Game(random=True)
        while game.status == ONGOING:
            move = self.choose_action(game, 0)
            game.make_move(move)
        return game.reward


def encode_states(states, result, original_rewards):
    new_df = pd.DataFrame(columns=['state', 'reward', 'original_reward'])
    rows = []
    for i in range(0, 22):
        row = {
            'state': i,
            'reward': result if i in states else 0,
            'original_reward': original_rewards[i]
        }
        rows.append(row)
    new_df = pd.concat([new_df, pd.DataFrame(rows)], ignore_index=True)
    new_df = new_df[['state', 'reward', 'original_reward']].values.astype('float32')
    return new_df


def encode_reward(X, reward):
    return np.array([reward] * len(X), dtype='float32').reshape(-1, 1)


def main():
    nn = NN()
    for i in range(1000):
        nn.train()
    statistics = {}
    epochs = 100
    for i in range(epochs):
        reward = nn.play()
        statistics[i] = reward
    print("Average reward:", sum(statistics.values()) / epochs)
    print("Loosed games:", len([x for x in statistics.values() if x == 0]))


if __name__ == "__main__":
    main()
