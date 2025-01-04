from random import randint
from consts import *
import numpy as np

class Game:
    def __init__(self, random=False):
        self.sum = 0
        self.reward = 0
        self.status = ONGOING
        self.random = random
        self.my_moves = []
        if random:
            # Initialize reward array with random values from 0 to 21
            self.rewards = np.random.randint(0, 22, size=22)
            print("Rewards:", self.rewards)

    def make_move(self, move: str):
        if move == "hit":
            # random number between 1 and 10
            new_card = randint(1, 10)
            self.sum += new_card
            print(f"New card: {new_card}")
            if self.random:
                print(f"Reward: {self.rewards[new_card]}")
                print("Your sum:", self.sum)
            else:
                print("Your sum:", self.sum)
            self.check_reward(new_card)
            self.my_moves.append(new_card)
        elif move == "stay":
            self.check_reward(self.my_moves[-1])
            self.status = 0
        else:
            print("Invalid move")

    def check_reward(self, new_card):
        if self.sum > JACKPOT:
            self.reward = 0
            self.status = 0
        else:
            if self.random:
                self.reward = self.rewards[new_card]
            else:
                self.reward = self.sum
        return self.reward


def main():
    # get from user if he wants to play with random rewards
    random = input("Do you want to play with random rewards? (Y/N): ")
    if random.lower() == "y":
        game = Game(random=True)
    else:
        game = Game()
    # get input from user
    while game.status == ONGOING:
        move = input("Enter move: ")
        game.make_move(move)
    if game.reward == 0:
        print("You lost!")
    else:
        print(f"You won {game.reward} coins!")


if __name__ == "__main__":
    main()