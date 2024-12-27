from random import randint
from consts import *


class Game:
    def __init__(self):
        self.sum = 0
        self.reward = 0
        self.status = ONGOING

    def make_move(self, move: str):
        if move == "hit":
            # random number between 1 and 10
            new_card = randint(1, 10)
            self.sum += new_card
            print(f"New card: {new_card}")
            print("Your sum:", self.sum)
            self.check_reward()
        elif move == "stay":
            self.check_reward()
            self.status = 0
        else:
            print("Invalid move")

    def check_reward(self):
        if self.sum > JACKPOT:
            self.reward = 0
            self.status = 0
        else:
            self.reward = self.sum
        return self.reward


def main():
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