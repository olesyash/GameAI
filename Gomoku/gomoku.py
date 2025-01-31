from base64 import encode
import numpy as np
import torch

ONGOING = -17
BLACK_WIN = 1
WHITE_WIN = -1
DRAW = 0
WIN_COUNT=4
BOARD_SIZE=7
BOARD_TENSOR = "board_tensor"
POLICY_PROBS = "policy_probs"
STATUS = "status"
CURRENT_PLAYER = "current_player"



class Gomoku:
    def __init__(self, board_size=BOARD_SIZE):
        """Initialize the Gomoku game.
        
        Args:
            board_size (int): Size of the board (default: 15x15)
        """
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1  # 1 for black, -1 for white
        self.move_history = []
        self.status = ONGOING  # -17 for ongoing, will be player number (1 or 2) when won, or 0 for draw
        self.last_move = None
        self.winner = None

    def reset(self):
        """Reset the game to initial state."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.move_history.clear()
        self.last_move = None
        self.status = ONGOING
        self.winner = None

    def legal_moves(self):
        """Get a list of legal moves.

        Returns:
            list: List of legal moves as (row, col) tuples
        """
        return [(r, c) for r in range(self.board_size) for c in range(self.board_size) if self.board[r, c] == 0]

    def is_legal_move(self, row, col):
        """Check if a move is legal.
        
        Args:
            row (int): Row index
            col (int): Column index
            
        Returns:
            bool: True if move is legal, False otherwise
        """
        # Check if position is within board bounds
        legal_moves = self.legal_moves()
        if (row, col) not in legal_moves:
            return False
        
        # Check if position is empty
        return self.board[row, col] == 0

    def make_move(self, row_col):
        """Make a move on the board.
        
        Args:
            row_col (tuple): Row and column indices
            
        Returns:
            bool: True if move was successful, False otherwise
        """
        row, col = row_col
        if not self.is_legal_move(row, col) or self.status != ONGOING:
            return False

        self.board[row, col] = self.current_player
        self.move_history.append((row, col))
        self.last_move = (row, col)

        # Check for win
        if self._check_win(row, col):
            self.status = self.current_player
            self.winner = self.current_player
        elif len(self.move_history) == self.board_size * self.board_size:
            self.status = 0  # Draw

        # Switch player
        self.current_player = 0 - self.current_player  # Switches between 1 and -1
        return True

    def unmake_move(self):
        """Undo the last move."""
        if not self.move_history:
            return False

        row, col = self.move_history.pop()
        self.last_move = self.move_history[-1] if self.move_history else None
        self.board[row, col] = 0
        self.current_player = 0 - self.current_player
        self.status = ONGOING
        self.winner = None
        return True

    def _check_win(self, row, col):
        """Check if the last move resulted in a win.
        
        Args:
            row (int): Row of last move
            col (int): Column of last move
            
        Returns:
            bool: True if the last move won the game, False otherwise
        """
        player = self.board[row, col]
        directions = [
            (1, 0),   # vertical
            (0, 1),   # horizontal
            (1, 1),   # diagonal
            (1, -1)   # anti-diagonal
        ]

        for dr, dc in directions:
            count = 1
            
            # Check in positive direction
            r, c = row + dr, col + dc
            while (0 <= r < self.board_size and 
                   0 <= c < self.board_size and 
                   self.board[r, c] == player):
                count += 1
                r += dr
                c += dc

            # Check in negative direction
            r, c = row - dr, col - dc
            while (0 <= r < self.board_size and 
                   0 <= c < self.board_size and 
                   self.board[r, c] == player):
                count += 1
                r -= dr
                c -= dc

            if count >= WIN_COUNT:
                return True

        return False

    def get_board(self):
        """Get the current board state.
        
        Returns:
            numpy.ndarray: Current board state
        """
        return self.board.copy()

    def get_current_player(self):
        """Get the current player.
        
        Returns:
            int: Current player (1 for black, 2 for white)
        """
        return self.current_player

    def is_game_over(self):
        """Check if the game is over.
        
        Returns:
            bool: True if game is over, False otherwise
        """
        return self.status != ONGOING

    def get_winner(self):
        """Get the winner of the game.
        
        Returns:
            int or None: Winner (1 for black, 2 for white, None if no winner)
        """
        return self.winner

    def clone(self):
        """Create a copy of the game.

        Returns:
            Gomoku: Copy of the game
        """
        clone = Gomoku(self.board_size)
        clone.board = self.board.copy()
        clone.current_player = self.current_player
        clone.move_history = self.move_history.copy()
        clone.last_move = self.last_move
        clone.status = self.status
        clone.winner = self.winner
        return clone

    def encode(self):
        """Encode the game state into policy probabilities and other features.
        
        Returns:
            dict: Contains board tensor, policy probabilities, and game state
        """
        # Convert board to tensor format
        board_tensor = torch.FloatTensor(self.board).view(1, 1, self.board_size, self.board_size)
        
        # Create policy probabilities (uniform over legal moves)
        policy_probs = np.zeros(self.board_size * self.board_size)
        legal_moves = self.legal_moves()
        if legal_moves:
            prob_per_move = 1.0 / len(legal_moves)
            for move in legal_moves:
                move_idx = move[0] * self.board_size + move[1]
                policy_probs[move_idx] = prob_per_move
        
        # Convert to tensors with correct shapes
        policy_probs = torch.FloatTensor(policy_probs)
        status_tensor = torch.FloatTensor([[self.status]])  # Shape: [1, 1]
        
        encoded_game = {}
        encoded_game[BOARD_TENSOR] = board_tensor
        encoded_game[CURRENT_PLAYER] = self.current_player
        encoded_game[STATUS] = status_tensor
        encoded_game[POLICY_PROBS] = policy_probs
        
        return encoded_game

    def decode(self, game_array):
        """ Decode the Game from vector"""
        self.board = game_array[:-2].reshape(self.board_size, self.board_size)
        self.current_player = int(game_array[-2])
        self.status = int(game_array[-1])
        self.winner = self.current_player if self.status == self.current_player else None

    def get_reward(self):
        """Returns the reward for the current game state."""
        if self.status == BLACK_WIN:
            return 1  # Reward for BLACK winning
        elif self.status == WHITE_WIN:
            return -1  # Reward for WHITE winning
        elif self.status == DRAW:
            return 0  # Reward for a draw
        else:
            # Game is ongoing; no reward yet
            return 0
