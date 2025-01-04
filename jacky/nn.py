from random import randint

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from consts import *
import numpy as np
from jacky import Game
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ReLU
from tensorflow.keras.optimizers import Adam

class NN():
    def __init__(self):
        # Initialize neural network with random weights
        self.Q = {}
        self.N = {}
        # Fill up the dictionary for all possible visits
        for i in range(0, JACKPOT + 1):
            for j in [HIT, STAY]:
                self.N[(i, j)] = 0
        print(self.N)

        # Set the learning rate
        learning_rate = 0.001
        optimizer = Adam(learning_rate=learning_rate)

        model = Sequential()
        model.add(ReLU(32))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.model = model

    def choose_action(self, game, epsilon):
        # In (1-epsilon) we will do the best action
        # In epsilon we will do a random action
        if game.sum == 0:
            return HIT

        if randint(0, 100) < epsilon * 100:
            # In epsilon we will do a random action
            return HIT if randint(0, 1) == 0 else STAY
        return HIT if randint(0, 1) == 0 else STAY
        # else:
        #     # In (1-epsilon) we will do the best action
        #     return self.best_action(game)


    def best_action(self, game):
        self.model.predict(np.array([game.my_moves]))

    def train(self, epochs=100, epsilon=0.1):
        statistics = {}
        for i in range(epochs):
            jacky_game = Game(random=True)
            while jacky_game.status == ONGOING:
                # Choose action
                action = self.choose_action(jacky_game, epsilon)
                print("Action chosen:", action)
                # make move
                jacky_game.make_move(action)

            # encode the statest to df
            df = encode_states(jacky_game.my_moves, jacky_game.reward, jacky_game.rewards)
            # train the model
            self.model.fit(df, jacky_game.reward, epochs=10, verbose=1)

        #     # Check status
        #     if jacky_game.reward == 0:
        #         print("You lost!")
        #     else:
        #         print(f"You won {jacky_game.reward} coins!")
        #     statistics[i] = jacky_game.reward
        # print(statistics)
        # print("Average reward:", sum(statistics.values()) / epochs)
        # print("Loosed games:", len([x for x in statistics.values() if x == 0]))



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
    return new_df


def main():
    nn = NN()
    nn.train()


if __name__ == "__main__":
    main()