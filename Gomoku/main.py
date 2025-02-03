from gomoku import Gomoku, BOARD_SIZE
from puct import PUCTPlayer
import torch


def train_model():
    # Initialize game and player
    game = Gomoku(board_size=BOARD_SIZE)
    player = PUCTPlayer(exploration_weight=1.0, game=game)
    
    # Training settings
    num_episodes = 100
    batch_size = 32
    learning_rate = 0.001
    
    # Start training
    print("Starting training...")
    player.train(num_episodes=num_episodes, 
                 batch_size=batch_size,
                 learning_rate=learning_rate)
    
    return player


def play_game(player1, player2=None):
    """Play a game between two players or against self"""
    game, winner = player1.play_game(opponent=player2)
    print(f"Game finished! Winner: {'Black' if winner == 1 else 'White' if winner == -1 else 'Draw'}")
    # print("Final board state:")
    # print_board(game.board)


def print_board(board):
    """Print the game board in a readable format"""
    symbols = {1: 'X', -1: 'O', 0: '.'}
    for row in board:
        print(' '.join(symbols[cell] for cell in row))
    print()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Train the model
    trained_player = train_model()
    
    # Play a game against itself
    print("\nPlaying a game (trained model against itself)...")
    play_game(trained_player)
