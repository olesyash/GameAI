import math
import random
import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARE_SIZE = 100
RADIUS = int(SQUARE_SIZE/2 - 5)

# Calculate screen dimensions
width = COLUMN_COUNT * SQUARE_SIZE
height = (ROW_COUNT + 1) * SQUARE_SIZE  # Extra row for piece preview
screen = pygame.display.set_mode((width, height))

# Colors
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Game Constants
ROWS = 6
COLS = 7

# Define the Connect Four game state
# Define the MCTS Node
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
                              exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
        )
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
            current_state.make(move)
        return current_state.get_reward()

    def backpropagate(self, node, reward):
        """Backpropagation phase: Update the node values and visits up the tree."""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    
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

def draw_board(screen, game):
    # Draw the game board
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            # Draw blue rectangle for each cell
            pygame.draw.rect(screen, BLUE, (c*SQUARE_SIZE, r*SQUARE_SIZE + SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            # Draw black circle for empty cells
            pygame.draw.circle(screen, BLACK, (int(c*SQUARE_SIZE + SQUARE_SIZE/2), 
                                            int(r*SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE/2)), RADIUS)
    
    # Draw the pieces
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if game.board[c][r] == game.RED:
                pygame.draw.circle(screen, RED, (int(c*SQUARE_SIZE + SQUARE_SIZE/2), 
                                               height - int(r*SQUARE_SIZE + SQUARE_SIZE/2)), RADIUS)
            elif game.board[c][r] == game.YELLOW:
                pygame.draw.circle(screen, YELLOW, (int(c*SQUARE_SIZE + SQUARE_SIZE/2), 
                                                  height - int(r*SQUARE_SIZE + SQUARE_SIZE/2)), RADIUS)
    pygame.display.update()

def main():
    pygame.display.set_caption('Connect 4 with AI')
    game = ConnectFour()
    mcts = MCTS(exploration_weight=1.4)
    
    # Initial board draw
    screen.fill(BLACK)
    draw_board(screen, game)
    pygame.display.update()
    
    # Game loop
    game_over = False
    
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            
            if event.type == pygame.MOUSEMOTION and game.player == game.RED:
                # Clear the top row
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARE_SIZE))
                # Draw the piece preview
                posx = event.pos[0]
                pygame.draw.circle(screen, RED, (posx, int(SQUARE_SIZE/2)), RADIUS)
                pygame.display.update()
            
            if event.type == pygame.MOUSEBUTTONDOWN and game.player == game.RED:
                # Human player move
                posx = event.pos[0]
                col = int(posx // SQUARE_SIZE)
                
                if col in game.legal_moves():
                    game.make(col)
                    screen.fill(BLACK)
                    draw_board(screen, game)
                    
                    if game.is_terminal():
                        game_over = True
                        break
                    
                    # AI move
                    if game.player == game.YELLOW:
                        pygame.time.wait(500)  # Small delay for better visualization
                        best_node = mcts.search(game.clone(), iterations=3000)
                        game.make(best_node.state.last_move)
                        screen.fill(BLACK)
                        draw_board(screen, game)
                        
                        if game.is_terminal():
                            game_over = True
                            break
    
    # Game over screen
    if game.is_terminal():
        font = pygame.font.Font(None, 75)
        if game.status == game.RED_WIN:
            text = font.render("You Win!", True, RED)
        elif game.status == game.YELLOW_WIN:
            text = font.render("AI Wins!", True, YELLOW)
        else:
            text = font.render("Draw!", True, BLUE)
        
        text_rect = text.get_rect(center=(width/2, height/2))
        screen.blit(text, text_rect)
        pygame.display.update()
        pygame.time.wait(3000)
    
    pygame.quit()


if __name__ == "__main__":
    main()
