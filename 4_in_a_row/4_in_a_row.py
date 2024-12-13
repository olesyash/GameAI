import pygame
import numpy as np
import sys
import math
import random

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE
size = (width, height)

class Node:
    def __init__(self, state, parent=None):
        self.state = state  # The game state
        self.parent = parent  # Parent node
        self.children = []  # Child nodes
        self.visits = 0  # Number of times this node was visited
        self.value = 0  # Total reward accumulated for this node

    def is_fully_expanded(self):
        """Checks if all possible moves have been expanded."""
        return len(self.children) == len(self.state.legal_moves())

    def best_child(self, exploration_weight=1):
        """Selects the best child node based on UCT (Upper Confidence Bound for Trees)."""
        best = max(
            self.children,
            key=lambda child: child.value / (child.visits + 1e-6) +
                              exploration_weight * math.sqrt(math.log(self.visits + 1)
                                                             / (child.visits + 1e-6)))
        return best


# Define the MCTS algorithm
class MCTS:
    def __init__(self, exploration_weight=1):
        self.exploration_weight = exploration_weight

    def search(self, initial_state, iterations=1000):
        """Performs MCTS to find the best move."""
        root = Node(initial_state)

        for _ in range(iterations):
            # 1. Selection: Traverse the tree using UCT until reaching a leaf node.
            node = self.select(root)

            # 2. Expansion: Add a new child node by exploring an unvisited move.
            if not node.state.is_terminal():
                node = self.expand(node)

            # 3. Simulation: Simulate a random game from the new node.
            reward = self.simulate(node)

            # 4. Backpropagation: Update the value and visit counts up the tree.
            self.backpropagate(node, reward)

        # Return the best child node (without exploration weight).
        return root.best_child(exploration_weight=0)

    def select(self, node):
        """Selection phase: Navigate the tree using UCT."""
        while not node.state.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
        return node

    def expand(self, node):
        """Expansion phase: Add a new child node for an unvisited move."""
        possible_moves = node.state.legal_moves()
        for move in possible_moves:
            if move not in [child.state.last_move for child in node.children]:
                new_state = node.state.clone()
                new_state.make(move)
                child_node = Node(new_state, parent=node)
                node.children.append(child_node)
                return child_node
        raise Exception("No moves to expand")


    def simulate(self, node):
        """Simulation phase: Play a random game until a terminal state is reached."""
        current_state = node.state.clone()  # Clone the state for simulation
        while not current_state.is_terminal():
            move = random.choice(current_state.legal_moves())
            print(move)
            current_state.make(move)
        return current_state.get_reward()

    def backpropagate(self, node, reward):
        """Backpropagation phase: Update the node values and visits up the tree."""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent


def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    # Check if the column is within bounds
    if col < 0 or col >= COLUMN_COUNT:
        return False
    # Check if the top cell in the column is empty
    return board[ROW_COUNT-1][col] == 0


def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r


def winning_move(board, piece):
    # Check horizontal locations
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Check vertical locations
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Check positively sloped diagonals
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Check negatively sloped diagonals
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True

    return False


def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
    
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):        
            if board[r][c] == 1:
                pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
            elif board[r][c] == 2:
                pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
    pygame.display.update()


def draw_new_game_button(screen, font):
    button_width = 200
    button_height = 50
    button_x = (COLUMN_COUNT * SQUARESIZE - button_width) // 2
    button_y = (ROW_COUNT + 1) * SQUARESIZE // 2
    
    # Draw button background
    pygame.draw.rect(screen, GREEN, (button_x, button_y, button_width, button_height))
    
    # Draw button text
    text = font.render("New Game", True, WHITE)
    text_rect = text.get_rect(center=(button_x + button_width//2, button_y + button_height//2))
    screen.blit(text, text_rect)
    
    return pygame.Rect(button_x, button_y, button_width, button_height)


def reset_game(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            board[r][c] = 0
    return 0, False  # Reset turn to 0 and game_over to False


def handle_player2_move(game, myfont, screen):
    mcts = MCTS(exploration_weight=1)
    best_node = mcts.search(game.clone(), iterations=1000)
    col = best_node.state.last_move  # Extract the move leading to the best state

    game_over = False
    row = get_next_open_row(game.board, col)
    drop_piece(game.board, row, col, 2)

    if winning_move(game.board, 2):
        label = myfont.render("Player 2 wins!!", 1, YELLOW)
        screen.blit(label, (40, 10))
        game_over = True

    return True, game_over


class ConnectFour:
    # Constants for the game board
    RED = 1
    YELLOW = -1
    EMPTY = 0

    # Game status constants
    RED_WIN = 1
    YELLOW_WIN = -1
    DRAW = 0
    ONGOING = -17  # Completely arbitrary

    def __init__(self):
        self.board = create_board()
        self.heights = [0 for _ in range(7)]  # The column heights in the board.
        self.player = self.RED
        self.status = self.ONGOING
        self.last_move = None  # Track the last move made

    def is_terminal(self):
        """Check if the game has ended in a win or draw."""
        # The game is terminal if there's a winner or the board is full (draw).
        return self.status in {self.RED_WIN, self.YELLOW_WIN, self.DRAW}

    def get_reward(self):
        """Returns the reward for the current game state."""
        if self.status == self.RED_WIN:
            return 1  # Reward for RED winning
        elif self.status == self.YELLOW_WIN:
            return -1  # Reward for YELLOW winning
        elif self.status == self.DRAW:
            return 0  # Reward for a draw
        else:
            # Game is ongoing; no reward yet
            return 0

    def legal_moves(self):
        return [i for i in range(7) if self.heights[i] < 6]

    def make(self, move):  # Assumes that 'move' is a legal move
        self.last_move = move
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
            while 0 <= x < 6 and 0 <= y < 7 and self.board[x][y] == player:
                count += 1
                x += dx
                y += dy
            x, y = col - dx, row - dy
            while 0 <= x < 6 and 0 <= y < 7 and self.board[x][y] == player:
                count += 1
                x -= dx
                y -= dy
            if count >= 3:
                return True
        return False

    def play(self):
        game_over = False
        turn = 0
        pygame.init()

        global screen # Make screen a global variable so it can be accessed in other functions
        screen = pygame.display.set_mode(size)
        pygame.display.set_caption('4 in a Row')
        myfont = pygame.font.SysFont("monospace", 75)
        button_font = pygame.font.SysFont("monospace", 36)  # Smaller font for button

        draw_board(self.board)
        pygame.display.update()

        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.MOUSEMOTION:
                    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                    posx = event.pos[0]
                    if turn == 0:
                        pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
                    else:
                        pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE/2)), RADIUS)
                    pygame.display.update()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                    # Ask for Player 1 Input
                    if turn == 0:
                        posx = event.pos[0]
                        col = int(math.floor(posx/SQUARESIZE))

                        if is_valid_location(self.board, col):
                            row = get_next_open_row(self.board, col)
                            drop_piece(self.board, row, col, 1)

                            if winning_move(self.board, 1):
                                label = myfont.render("Player 1 wins!!", 1, RED)
                                screen.blit(label, (40, 10))
                                game_over = True

                            draw_board(self.board)
                            # Only change turns if the move was valid
                            turn += 1
                            turn = turn % 2

            # Player 2 (Computer) makes a move automatically when it's their turn
            if not game_over and turn == 1:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                move_made, game_over = handle_player2_move(self, myfont, screen)
                draw_board(self.board)
                turn += 1
                turn = turn % 2
                pygame.time.wait(1000)  # Add a small delay to make the computer's moves visible

            if game_over:
                # Draw the new game button
                button_rect = draw_new_game_button(screen, button_font)
                pygame.display.update()

                # Wait for either quit or new game button click
                waiting_for_input = True
                while waiting_for_input:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            sys.exit()
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            mouse_pos = event.pos
                            if button_rect.collidepoint(mouse_pos):
                                # Reset the game
                                turn, game_over = reset_game(self.board)
                                pygame.draw.rect(screen, BLACK, (0, 0, width, height))
                                draw_board(self.board)
                                waiting_for_input = False
                                break

def main():
    game = ConnectFour()
    game.play()


if __name__ == "__main__":
    main()
