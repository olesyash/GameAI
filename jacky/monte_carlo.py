from random import randint

from consts import *
from jacky import Game


class MCTS:
    def __init__(self):
        # {state : action}
        self.Q = {}
        # Fill up the dictionary for all possible states
        for i in range(0, JACKPOT + 1):
            for j in ["hit", "stay"]:
                self.Q[(i, j)] = 0
        print(self.Q)
        self.N = {}
        # Fill up the dictionary for all possible visits
        for i in range(0, JACKPOT + 1):
            for j in ["hit", "stay"]:
                self.N[(i, j)] = 0
        print(self.N)

    def choose_action(self, game):
        if game.sum == 0:
            return "hit"

        # choose randomly between hit and stay
        if randint(0, 1) == 0:
            return "hit"
        else:
            return "stay"

    def update(self, prev_sum, game, action):
        if prev_sum > JACKPOT:
            return
        self.N[(prev_sum, action)] += 1
        print(self.N)
        self.Q[(prev_sum, action)] = self.Q[(prev_sum, action)] + (game.reward - self.Q[(prev_sum, action)]) / self.N[(prev_sum, action)]
        print(self.Q)

    def train(self):
        jacky_game = Game()
        moves = []
        while jacky_game.status == ONGOING:
            # Choose action
            action = self.choose_action(jacky_game)
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


mcts = MCTS()

iterations = 10
while iterations > 0:
    mcts.train()
    iterations -= 1



