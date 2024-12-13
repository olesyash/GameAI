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

class MCTSNode:
    def __init__(self):
        self.visits = 0
        self.value = 0
        self.children = {}
        self.parent = None


class MCTSPlayer:
    def __init__(self, iterations=10):
        self.iterations = iterations

    def choose_move(self, board):
        valid_move = False
        while not valid_move:
            col = random.randint(0, COLUMN_COUNT - 1)
            valid_move = is_valid_location(board, col)
        return col

    def update_rule(self, node, reward):
        """
        Update the node's value and visit count using the formula Q = Q + 1/n * (r - Q)
        :param node: The node to update
        :param reward: The reward received
        """
        # Update the node's value and visit count
        node.visits += 1
        node.value += (reward - node.value) / node.visits




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


def handle_player2_move(board, event, myfont, screen):
    player = MCTSPlayer()
    col = player.choose_move(board)
    
    game_over = False
    row = get_next_open_row(board, col)
    drop_piece(board, row, col, 2)

    if winning_move(board, 2):
        label = myfont.render("Player 2 wins!!", 1, YELLOW)
        screen.blit(label, (40, 10))
        game_over = True

    return True, game_over


class ConnectFour:
    def __init__(self):
        self.board = create_board()

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
                move_made, game_over = handle_player2_move(self.board, None, myfont, screen)
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
