from base64 import encode
import numpy as np
import torch
import sys

ONGOING = -17
BLACK_WIN = BLACK = 1
WHITE_WIN = WHITE = -1
DRAW = 0
WIN_COUNT = 3
BOARD_SIZE = 3
BOARD_TENSOR = "board_tensor"
POLICY_PROBS = "policy_probs"
STATUS = "status"
CURRENT_PLAYER = "current_player"
IS_ONGOING = "is_ongoing"
VALUE = "value"


class Gomoku:
    def __init__(self, board_size=BOARD_SIZE):
        """Initialize the Gomoku game.
        
        Args:
            board_size (int): Size of the board (default: 15x15)
        """
        self.board_size = board_size
        self.win_count = WIN_COUNT
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.next_player = 1  # 1 for black, -1 for white
        self.move_history = []
        self.status = ONGOING  # -17 for ongoing, will be player number (1 or 2) when won, or 0 for draw
        self.last_move = None
        self.winner = None

    def reset(self):
        """Reset the game to initial state."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.next_player = 1
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

        self.board[row, col] = self.next_player
        self.move_history.append(((row, col), self.next_player))
        self.last_move = (row, col)

        # Check for win
        if self._check_win(row, col):
            self.status = self.next_player
            self.winner = self.next_player
        elif len(self.move_history) == self.board_size * self.board_size:
            self.status = 0  # Draw
            self.winner = 0  # Draw
        self.switch_player()       
        return True

    def switch_player(self):
        """Switch the current player."""
        self.next_player = 0 - self.next_player

    def unmake_move(self):
        """Undo the last move."""
        if not self.move_history:
            return False

        (row, col), player = self.move_history.pop()
        self.last_move = self.move_history[-1][0:2] if self.move_history else None
        self.board[row, col] = 0
        self.switch_player()
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
            int: Current player (1 for black, -1 for white)
        """
        return 0 - self.next_player

    def is_game_over(self):
        """Check if the game is over.
        
        Returns:
            bool: True if game is over, False otherwise
        """
        return self.status != ONGOING

    def print_board(self):
        print(self.board)
        sys.stdout.flush()
        sys.stderr.flush()

    def get_winner(self):
        """Get the winner of the game.
        
        Returns:
            int or None: Winner (1 for black, -1 for white, None if no winner)
        """
        return self.winner

    def clone(self):
        """Create a copy of the game.

        Returns:
            Gomoku: Copy of the game
        """
        clone = Gomoku(self.board_size)
        clone.board = self.board.copy()
        clone.next_player = self.next_player
        clone.move_history = self.move_history.copy()
        clone.last_move = self.last_move
        clone.status = self.status
        clone.winner = self.winner
        return clone

    def compute_threat_matrix(self, board):
        """ Compute a threat matrix based on the current board position.
        
        Args:
            board (np.array): The current board state (size NxN)
        
        Returns:
            np.array: Threat matrix of the same size as the board, with threat values.
        """
        board_size = board.shape[0]
        threat_matrix = np.zeros((board_size, board_size))

        # Scan every cell on the board
        for r in range(board_size):
            for c in range(board_size):
                if board[r, c] != 0:
                    continue  # Skip occupied positions

                # Check in four possible directions (horizontal, vertical, two diagonals)
                for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                    count = 0
                    open_ends = 0
                    
                    # Look forward
                    for i in range(1, 5):
                        nr, nc = r + i * dr, c + i * dc
                        if 0 <= nr < board_size and 0 <= nc < board_size:
                            if board[nr, nc] == 1:  # Black stone
                                count += 1
                            elif board[nr, nc] == 0:  # Empty space
                                open_ends += 1
                            else:
                                break  # Blocked by opponent

                    # Look backward
                    for i in range(1, 5):
                        nr, nc = r - i * dr, c - i * dc
                        if 0 <= nr < board_size and 0 <= nc < board_size:
                            if board[nr, nc] == 1:
                                count += 1
                            elif board[nr, nc] == 0:
                                open_ends += 1
                            else:
                                break  # Blocked by opponent

                    # Assign threat scores based on patterns
                    if count == 4 and open_ends > 0:
                        threat_matrix[r, c] += 4  # Open Four - very strong
                    elif count == 3 and open_ends > 1:
                        threat_matrix[r, c] += 3  # Open Three - strong
                    elif count == 3 and open_ends == 1:
                        threat_matrix[r, c] += 2  # Closed Three - moderate
                    elif count == 2 and open_ends > 1:
                        threat_matrix[r, c] += 1  # Two in a row - potential move

        return threat_matrix

    def encode(self, n_history=3):
        """Encode the game state for neural network input.
        
        Args:
            n_history (int): Number of past moves to include for each player
        
        Returns:
            torch.Tensor: Multi-channel tensor representing the board state:
                - Channel 1: Current player's moves (1 where current player has moved, 0 elsewhere)
                - Channel 2: Opponent's moves (1 where opponent has moved, 0 elsewhere)
                - Channels 3 to N+2: Last N moves of current player (most recent first)
                - Channels N+3 to 2N+2: Last N moves of opponent (most recent first)
                - Channel 2N+3: Current player indicator (all 1s if Black's turn, all 0s if White's turn)
        """
        board_size = len(self.board)
        current_player = self.next_player
        
        # Create channels with proper dimensions: 2 for current state + 2N for move history + 1 for player indicator
        channels = torch.zeros((2 * n_history + 3, board_size, board_size))
        
        # Fill first two channels (current player and opponent positions)
        for i in range(board_size):
            for j in range(board_size):
                if self.board[i][j] == current_player:
                    channels[0, i, j] = 1
                elif self.board[i][j] == -current_player:
                    channels[1, i, j] = 1
        
        # Add move history channels
        if hasattr(self, 'move_history') and self.move_history:
            current_moves = []
            opponent_moves = []
            
            # Separate moves by player
            for move, player in reversed(self.move_history):
                if player == current_player:
                    current_moves.append(move)
                else:
                    opponent_moves.append(move)
            
            # Fill current player's move history
            for i, move in enumerate(current_moves[:n_history]):
                x, y = move
                channels[2 + i, x, y] = 1
            
            # Fill opponent's move history
            for i, move in enumerate(opponent_moves[:n_history]):
                x, y = move
                channels[2 + n_history + i, x, y] = 1
        
        # Add current player indicator channel (1s for Black, 0s for White)
        if current_player == BLACK:  # BLACK = 1
            channels[-1] = torch.ones((board_size, board_size))
        
        return channels

    def decode(self, value):
        """ Decode the value from vector(-1, 1) to actual value 0,1,-1
            Use thresholds to determine the value
        """
        if value > 0.3:  # Strong positive -> 1
            return 1
        elif value < -0.3:  # Strong negative -> -1
            return -1
        else:  # Values close to zero -> 0
            return 0

    def __eq__(self, other):
        """Compare two Gomoku game states for equality.
        
        Two games are equal if they have:
        1. Same board size
        2. Same board state (all positions match)
        3. Same move history
        """
        if not isinstance(other, Gomoku):
            return False
            
        # Check board size matches
        if self.board_size != other.board_size:
            return False
            
        # Check board states match
        if not np.array_equal(self.board, other.board):
            return False
            
        # Check move histories match
        if self.move_history != other.move_history:
            return False
            
        return True
        
    def __hash__(self):
        """Generate hash for Gomoku game state.
        
        This allows Gomoku objects to be used in sets and as dict keys.
        Hash is based on board state and move history.
        """
        # Convert board to tuple for hashing
        board_tuple = tuple(map(tuple, self.board))
        # Convert move history to tuple for hashing
        moves_tuple = tuple(map(tuple, self.move_history))
        
        return hash((board_tuple, moves_tuple, self.board_size))
