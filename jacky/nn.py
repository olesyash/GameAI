from random import randint
from uu import encode

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from consts import *
import numpy as np
from jacky import Game
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU, Input
from tensorflow.keras.optimizers import Adam

class NN():
    def __init__(self):
        # Set the learning rate and create optimizer with decay
        learning_rate = 0.001
        optimizer = Adam(learning_rate=learning_rate)

        # Create a deeper network
        model = Sequential([
            Input(shape=(66,)),
            Dense(64),
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
        Choose the best action using Q-learning principles:
        - For each action, estimate Q(s,a) = immediate_reward + gamma * future_reward
        - Choose the action with highest Q-value
        """
        gamma = 0.95  # discount factor for future rewards
        
        # Estimate Q-value for STAY
        stay_reward = game.check_reward()
        q_stay = stay_reward / JACKPOT  # normalized reward
        
        # Estimate Q-value for HIT
        possible_next_cards = list(range(1, 11))  # cards 1-10
        q_hit = 0
        valid_futures = 0
        
        # Simulate possible next states
        for card in possible_next_cards:
            next_sum = game.sum + card
            if next_sum <= JACKPOT:  # only consider valid future states
                valid_futures += 1
                immediate_reward = game.rewards[next_sum] / JACKPOT if game.random else next_sum / JACKPOT
                # Get future value estimation
                future_state = encode_states([next_sum], immediate_reward, game.rewards)
                future_value = self.model.predict(future_state, verbose=0)[0][0]
                q_hit += (immediate_reward + gamma * future_value) / 10  # average over possible cards
        
        if valid_futures == 0:  # if no valid future states
            return STAY
            
        # Choose action with highest Q-value
        return HIT if q_hit > q_stay else STAY

    def train(self, epochs=5, epsilon=0.1):
        for i in range(epochs):
            print("Epoch:", i)
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

            # Save model
            self.model.save("model.keras")
            


    def play(self):
        game = Game(random=True)
        while game.status == ONGOING:
            move = self.choose_action(game, 0)
            game.make_move(move)
        return game.reward


def encode_states(states, result, original_rewards):
    encoded = []
    # First add all states
    for i in range(0, 22):
        encoded.append(i / JACKPOT)
    # Then add all results
    for i in range(0, 22):
        encoded.append(result / JACKPOT if i in states else 0)
    # Finally add all original rewards
    for i in range(0, 22):
        encoded.append(original_rewards[i] / JACKPOT)
    return np.array(encoded, dtype='float32').reshape(1, -1)


def encode_reward(X, reward):
    return np.array([reward / JACKPOT] * len(X), dtype='float32').reshape(-1, 1)


def main():
    to_train = False
    if to_train:
        print("Training...")
        nn = NN()
        # try load the model
        try:
            nn.model.load_weights("model.keras")
        except:
            print("Model not found")
        
        total_episodes = 10000
        eval_frequency = 100
        eval_episodes = 100  # number of episodes to run during evaluation
        best_avg_reward = float('-inf')
        
        for episode in range(total_episodes):
            # Training
            nn.train(epochs=1, epsilon=0.1)
            
            # Evaluation every eval_frequency episodes
            if (episode + 1) % eval_frequency == 0:
                print(f"\nEvaluation after episode {episode + 1}:")
                rewards = []
                for _ in range(eval_episodes):
                    reward = nn.play()
                    rewards.append(reward)
                
                avg_reward = sum(rewards) / len(rewards)
                lost_games = len([r for r in rewards if r == 0])
                win_rate = (len(rewards) - lost_games) / len(rewards) * 100
                
                print(f"Average reward: {avg_reward:.2f}")
                print(f"Win rate: {win_rate:.1f}%")
                print(f"Lost games: {lost_games}/{len(rewards)}\n")
                
                # Save model only if it's the best so far
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    nn.model.save("best_model.keras")
                    print(f"New best model saved with average reward: {avg_reward:.2f}")
    else:
        print("Playing...")
        nn = NN()
        nn.model.load_weights("best_model.keras")
        rewards = []
        eval_episodes = 100
        for _ in range(eval_episodes):
            reward = nn.play()
            rewards.append(reward)
        
        avg_reward = sum(rewards) / len(rewards)
        lost_games = len([r for r in rewards if r == 0])
        win_rate = (len(rewards) - lost_games) / len(rewards) * 100
        
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Lost games: {lost_games}/{len(rewards)}\n")
      

if __name__ == "__main__":
    main()
