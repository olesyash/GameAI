from random import randint

from consts import *
from jacky import Game


class MCTS:
    def __init__(self):
        # {state : action}
        self.Q = {}
        # Fill up the dictionary for all possible states
        for i in range(0, JACKPOT + 1):
            for j in [HIT, STAY]:
                self.Q[(i, j)] = 0
        print(self.Q)
        self.N = {}
        # Fill up the dictionary for all possible visits
        for i in range(0, JACKPOT + 1):
            for j in [HIT, STAY]:
                self.N[(i, j)] = 0
        print(self.N)

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
        best_action = HIT
        best_value = self.Q[(game.sum, HIT)]
        if self.Q[(game.sum, STAY)] > best_value:
            best_action = STAY
            best_value = self.Q[(game.sum, STAY)]
        return best_action

    def update(self, prev_sum, game, action):
        if prev_sum > JACKPOT:
            return
        self.N[(prev_sum, action)] += 1
        print(self.N)
        self.Q[(prev_sum, action)] = self.Q[(prev_sum, action)] + (game.reward - self.Q[(prev_sum, action)]) / self.N[(prev_sum, action)]
        print(self.Q)

    def play(self):
        game = Game()
        while game.status == ONGOING:
            move = self.choose_action(game, 0)
            game.make_move(move)
        return game.reward

    def train(self, epochs=100, epsilon=0.1):
        statistics = {}
        for i in range(epochs):
            jacky_game = Game()
            moves = []
            while jacky_game.status == ONGOING:
                # Choose action
                action = self.choose_action(jacky_game, epsilon)
                prev_sum = jacky_game.sum
                print("Action chosen:", action)
                # make move
                jacky_game.make_move(action)
                moves.append((prev_sum, action))
            # update the Q value
            for prev_sum, action in moves:
                self.update(prev_sum, jacky_game, action)
            # Check status
            if jacky_game.reward == 0:
                print("You lost!")
            else:
                print(f"You won {jacky_game.reward} coins!")
            statistics[i] = jacky_game.reward
        print(statistics)
        print("Average reward:", sum(statistics.values()) / epochs)
        print("Loosed games:", len([x for x in statistics.values() if x == 0]))

mcts = MCTS()

mcts.train(epochs=1000, epsilon=0.1)

statistics = {}
epochs = 100
for i in range(epochs):
    reward = mcts.play()
    statistics[i] = reward
print("Average reward:", sum(statistics.values()) / epochs)
print("Loosed games:", len([x for x in statistics.values() if x == 0]))



