import tkinter as tk
from tkinter import messagebox
from gomoku import Gomoku
from MCTS import MCTSPlayer
from puct import PUCTPlayer,PUCTNode
from gomoku import ONGOING


PUCT = 'puct'
MCTS = 'mcts'
MANUAL = 'manual'


class GomokuGUI:
    def __init__(self, master, game_mode=MANUAL):
        self.master = master
        self.master.title("Gomoku")
        self.game = Gomoku()
        self.game_mode = game_mode
        
        # Initialize players based on game mode
        if game_mode == MCTS:
            self.mcts = MCTSPlayer(exploration_weight=2)
        elif game_mode == PUCT:
            self.puct_player = PUCTPlayer(exploration_weight=1.4, game=self.game)

        self.cell_size = 40
        self.padding = 20
        self.ai_thinking = False

        # Calculate canvas size based on board size and padding
        canvas_size = self.game.board_size * self.cell_size + 2 * self.padding
        
        # Create canvas
        self.canvas = tk.Canvas(
            master, 
            width=canvas_size,
            height=canvas_size,
            bg='#DEB887'  # Burlywood color for board
        )
        self.canvas.pack(padx=10, pady=10)
        
        # Draw board
        self.draw_board()
        
        # Bind click event
        self.canvas.bind('<Button-1>', self.handle_click)
        
        # Create control buttons
        self.create_controls()

    def create_controls(self):
        """Create control buttons."""
        control_frame = tk.Frame(self.master)
        control_frame.pack(pady=5)
        
        # Reset button
        reset_btn = tk.Button(
            control_frame,
            text="New Game",
            command=self.reset_game
        )
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Undo button
        undo_btn = tk.Button(
            control_frame,
            text="Undo",
            command=self.undo_move
        )
        undo_btn.pack(side=tk.LEFT, padx=5)

    def draw_board(self):
        """Draw the game board grid."""
        # Clear canvas
        self.canvas.delete("all")
        
        # Draw grid lines
        for i in range(self.game.board_size):
            # Vertical lines
            x = self.padding + i * self.cell_size
            self.canvas.create_line(
                x, self.padding,
                x, self.game.board_size * self.cell_size + self.padding,
                fill='black'
            )
            
            # Horizontal lines
            y = self.padding + i * self.cell_size
            self.canvas.create_line(
                self.padding, y,
                self.game.board_size * self.cell_size + self.padding, y,
                fill='black'
            )
        
        # Draw stones
        board = self.game.get_board()
        for row in range(self.game.board_size):
            for col in range(self.game.board_size):
                if board[row, col] != 0:
                    self.draw_stone(row, col, board[row, col])

    def draw_stone(self, row, col, player):
        """Draw a stone on the board."""
        x = self.padding + col * self.cell_size
        y = self.padding + row * self.cell_size
        radius = self.cell_size // 2 - 2
        
        color = 'black' if player == 1 else 'white'
        outline = 'white' if player == 1 else 'black'
        
        self.canvas.create_oval(
            x - radius, y - radius,
            x + radius, y + radius,
            fill=color,
            outline=outline
        )

    def show_winner_box(self):
        winner = self.game.get_winner()
        if winner:
            color = "Black" if winner == 1 else "White"
            messagebox.showinfo("Game Over", f"{color} wins!")
        else:
            messagebox.showinfo("Game Over", "It's a draw!")
        
        # Print board state before showing dialog
        self.game.print_board()
        self.master.update()  # Force GUI update

    def handle_click(self, event):
        """Handle mouse click event."""
        # Block all clicks during AI turn
        if self.ai_thinking:
            return

        # Block if game is over
        if self.game.is_game_over():
            return
            
        # Convert click coordinates to board position
        col = round((event.x - self.padding) / self.cell_size)
        row = round((event.y - self.padding) / self.cell_size)
        
        # Validate move is within board boundaries
        if not (0 <= row < self.game.board_size and 0 <= col < self.game.board_size):
            return

        # Try to make the move
        if not self.game.make_move((row, col)):
            return

        # Move was successful, update display
        self.draw_board()
        self.master.update()

        if self.game.is_game_over():
            self.show_winner_box()
            return

        # AI move for MCTS or PUCT modes
        if self.game_mode in [MCTS, PUCT] and self.game.status == ONGOING:
            self.ai_thinking = True
            self.canvas.config(cursor="watch")
            self.master.update()
            self.execute_ai_move()
        else:
            return

    def execute_ai_move(self):
        """Execute AI move based on selected game mode."""
        # Perform AI move
        # MCTS
        if self.game_mode == MCTS:
            best_node, _ = self.mcts.search(self.game.clone(), iterations=7000)
            if self.game.make_move(best_node.state.last_move):
                self.draw_board()
        # PUCT
        elif self.game_mode == PUCT:
            state, _ = self.puct_player.best_move(self.game.clone(), iterations=7000)
            move = state.last_move
            if self.game.make_move(move):
                self.draw_board()

        self.ai_thinking = False
        self.canvas.config(cursor="arrow")

        # Check game status after AI move
        if self.game.is_game_over():
            self.show_winner_box()
            return

    def reset_game(self):
        """Reset the game."""
        # Reset game state
        self.game.reset()
        
        # Redraw the board
        self.draw_board()
        
        # Reinitialize AI player if needed
        if self.game_mode == MCTS:
            self.ai_player = MCTSPlayer(exploration_weight=1.4)
        elif self.game_mode == PUCT:
            self.ai_player = PUCTPlayer(exploration_weight=1.4, game=self.game)

    def undo_move(self):
        """Undo the last move."""
        if self.game.unmake_move():
            self.draw_board()

    @staticmethod
    def select_game_mode():
        """Create a window to select game mode."""
        root = tk.Tk()
        root.title("Gomoku - Game Mode Selection")
        root.geometry("300x200")
        
        mode_var = tk.StringVar(value="manual")
        result = {"mode": "manual"}  # Using a dict to store the result
        
        def on_submit():
            result["mode"] = mode_var.get()  # Store the selected mode
            root.quit()

        label = tk.Label(root, text="Select Game Mode", font=("Arial", 14))
        label.pack(pady=10)

        modes = [
            ("Manual Game", MANUAL),
            ("MCTS Player", MCTS),
            ("PUCT Player", PUCT)
        ]

        for text, mode in modes:
            rb = tk.Radiobutton(root, text=text, variable=mode_var, value=mode)
            rb.pack(pady=5)

        start_button = tk.Button(root, text="Start Game", command=on_submit)
        start_button.pack(pady=10)

        root.mainloop()
        selected_mode = result["mode"]  # Get the stored mode
        root.destroy()
        return selected_mode


def main():
    """Main game method with mode selection."""
    game_mode = GomokuGUI.select_game_mode()
    root = tk.Tk()
    app = GomokuGUI(root, game_mode)
    root.mainloop()


if __name__ == "__main__":
    main()
