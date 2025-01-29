import tkinter as tk
from tkinter import messagebox
from gomoku import Gomoku
from MCTS import MCTS


class GomokuGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Gomoku")
        self.game = Gomoku()
        self.mcts = MCTS(exploration_weight=1.4)
        self.cell_size = 40
        self.padding = 20
        
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

    def handle_click(self, event):
        """Handle mouse click event."""
        if self.game.is_game_over():
            return
            
        # Convert click coordinates to board position
        col = round((event.x - self.padding) / self.cell_size)
        row = round((event.y - self.padding) / self.cell_size)
        
        if self.game.make_move((row, col)):
            self.draw_board()
            self.master.update()  # Force GUI update
            
            if self.game.is_game_over():
                self.show_winner_box()
            else:
                # AI turn
                self.ai_turn()

    def ai_turn(self):
        print("Board state before AI move: ")
        print(str(self.game.board))
        best_node = self.mcts.search(self.game.clone(), iterations=50000)
        if self.game.make_move(best_node.state.last_move):
            print("Board state after AI move: ")
            print(str(self.game.board))
            self.draw_board()
        if self.game.is_game_over():
            self.show_winner_box()

    def reset_game(self):
        """Reset the game."""
        self.game.reset()
        self.draw_board()

    def undo_move(self):
        """Undo the last move."""
        if self.game.unmake_move():
            self.draw_board()


def main():
    root = tk.Tk()
    app = GomokuGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
