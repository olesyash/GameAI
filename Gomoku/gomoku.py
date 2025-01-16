import numpy as np

class Gomoku:
    def __init__(self, board_size=15):
        """Initialize the Gomoku game.
        
        Args:
            board_size (int): Size of the board (default: 15x15)
        """
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1  # 1 for black, 2 for white
        self.move_history = []
        self.game_over = False
        self.winner = None

    def reset(self):
        """Reset the game to initial state."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.move_history.clear()
        self.game_over = False
        self.winner = None

    def is_legal_move(self, row, col):
        """Check if a move is legal.
        
        Args:
            row (int): Row index
            col (int): Column index
            
        Returns:
            bool: True if move is legal, False otherwise
        """
        # Check if position is within board bounds
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False
        
        # Check if position is empty
        return self.board[row, col] == 0

    def make_move(self, row, col):
        """Make a move on the board.
        
        Args:
            row (int): Row index
            col (int): Column index
            
        Returns:
            bool: True if move was successful, False otherwise
        """
        if not self.is_legal_move(row, col) or self.game_over:
            return False

        self.board[row, col] = self.current_player
        self.move_history.append((row, col))

        # Check for win
        if self._check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        elif len(self.move_history) == self.board_size * self.board_size:
            self.game_over = True  # Draw

        # Switch player
        self.current_player = 3 - self.current_player  # Switches between 1 and 2
        return True

    def unmake_move(self):
        """Undo the last move."""
        if not self.move_history:
            return False

        row, col = self.move_history.pop()
        self.board[row, col] = 0
        self.current_player = 3 - self.current_player
        self.game_over = False
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

            if count >= 5:
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
        return self.game_over

    def get_winner(self):
        """Get the winner of the game.
        
        Returns:
            int or None: Winner (1 for black, 2 for white, None if no winner)
        """
        return self.winner