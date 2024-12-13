
class ConnectFour:

    # Some constants that I want to use to fill the board.
    RED = 1
    YELLOW = -1
    EMPTY = 0

    # Game status constants
    RED_WIN = 1
    YELLOW_WIN = -1
    DRAW = 0
    ONGOING = -17  # Completely arbitrary

    def __init__(self):
        self.board = [[self.EMPTY for _ in range(6)] for _ in range(7)]
        self.heights = [0 for _ in range(7)]  # The column heights in the board.
        self.player = self.RED
        self.status = self.ONGOING

    def legal_moves(self):
        return [i for i in range(7) if self.heights[i] < 6]

    def make(self, move):  # Assumes that 'move' is a legal move
        self.board[move][self.heights[move]] = self.player
        self.heights[move] += 1
        # Check if the current move results in a winner:
        if self.winning_move(move):
            self.status = self.player
        elif len(self.legal_moves()) == 0:
            self.status = self.DRAW
        else:
            self.player = self.other(self.player)

    def other(self, player):
        return self.RED if player == self.YELLOW else self.YELLOW

    def unmake(self, move):
        self.heights[move] -= 1
        self.board[move][self.heights[move]] = self.EMPTY
        self.player = self.other(self.player)
        self.status = self.ONGOING

    def clone(self):
        clone = ConnectFour()
        clone.board = [col[:] for col in self.board]  # Deep copy columns
        clone.heights = self.heights[:]  # Deep copy heights
        clone.player = self.player
        clone.status = self.status
        return clone

    def winning_move(self, move):
        # Checks if the move that was just made wins the game.
        # Assumes that the player who made the move is still the current player.

        col = move
        row = self.heights[col] - 1  # Row of the last placed piece
        player = self.board[col][row]  # Current player's piece

        # Check all four directions: horizontal, vertical, and two diagonals
        # Directions: (dx, dy) pairs for all 4 possible win directions
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]   
        for dx, dy in directions:
            count = 0
            x, y = col + dx, row + dy
            while 0 <= x < 7 and 0 <= y < 6 and self.board[x][y] == player:
                count += 1
                x += dx
                y += dy
            x, y = col - dx, row - dy
            while 0 <= x < 7 and 0 <= y < 6 and self.board[x][y] == player:
                count += 1
                x -= dx
                y -= dy
            if count >= 3:
                return True
        return False

    def __str__(self):
        """
        Returns a string representation of the board.
        'R' for RED, 'Y' for YELLOW, '.' for EMPTY.
        """
        rows = []
        for r in range(5, -1, -1):  # From top row to bottom
            row = []
            for c in range(7):
                if self.board[c][r] == self.RED:
                    row.append("R")
                elif self.board[c][r] == self.YELLOW:
                    row.append("Y")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)

# Main function for debugging purposes: Allows two humans to play the game.
def main():
    """
    A simple main function for debugging purposes. 
    Allows two human players to play Connect Four in the terminal.
    """
    game = ConnectFour()

    print("Welcome to Connect Four!")
    print("Player 1 is RED (R) and Player 2 is YELLOW (Y).\n")

    while game.status == game.ONGOING:
        print(game)
        print("\nCurrent Player:", "RED" if game.player == game.RED else "YELLOW")
        try:
            move = int(input("Enter a column (0-6): "))
            if move not in game.legal_moves():
                print("Illegal move. Try again.")
                continue
            game.make(move)
        except ValueError:
            print("Invalid input. Enter a number between 0 and 6.")
        except IndexError:
            print("Move out of bounds. Try again.")

    print(game)
    if game.status == game.RED:
        print("\nRED (Player 1) wins!")
    elif game.status == game.YELLOW:
        print("\nYELLOW (Player 2) wins!")
    else:
        print("\nIt's a draw!")

if __name__ == "__main__":
    main()
