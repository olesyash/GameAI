from puct import PUCTPlayer
import torch
import numpy as np
from nn import GameNetwork
from gomoku import BOARD_SIZE, Gomoku
import matplotlib.pyplot as plt
import os
from MCTS import MCTSPlayer


def train_model():
    # Initialize game and network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    network = GameNetwork(BOARD_SIZE, device)
    network.to(device)  # Ensure the model is on the correct device
    
    # Training parameters
    exploration_weight = 1.0
    learning_rate = 0.001
    num_episodes = 1
    losses = []
    
    # Store all training data
    all_states = []
    all_policies = []
    all_values = []
    
    for episode in range(num_episodes):
        game = Gomoku(board_size=BOARD_SIZE)
        states_this_game = []  # Store all states in this game
        policies_this_game = []  # Store MCTS policies for each state
        
        # Create PUCT player with current network
        puct = PUCTPlayer(exploration_weight, game, fake_network=True)
        # Create MCTS player
        mcts = MCTSPlayer(exploration_weight)
        
        # Play one game
        while not game.is_game_over():
            # Get move probabilities from PUCT
            state, best_node = puct.best_move(game, iterations=800)
            move = state.last_move
            game.make_move(move)

            # Get move from MCTS
            best_node, root = mcts.search(game, iterations=800)

            current_state = game.clone()
            states_this_game.append(current_state)
            
            # Calculate policy from MCTS visit counts
            policy = np.zeros(BOARD_SIZE * BOARD_SIZE)
            total_visits = sum(child.visits for child in root.children)
            
            if total_visits > 0:
                for child in root.children:
                    move = child.state.last_move
                    move_idx = move[0] * BOARD_SIZE + move[1]
                    policy[move_idx] = child.visits / total_visits
            
            policies_this_game.append(policy)
            
            # Make the move
            move = best_node.state.last_move
            game.make_move(move)
       
        # Get game result
        winner = game.get_winner()
        print(f"Game finished! Winner: {'Black' if winner == 1 else 'White' if winner == -1 else 'Draw'}", flush=True)
        
        # Calculate values for all states in this game
        for state in states_this_game:
            player = state.next_player
            if winner == 0:
                value = 0  # Draw
            else:
                value = 1 if winner == player else -1
            all_values.append(value)
        
        # Add states and policies to our training data
        all_states.extend(states_this_game)
        all_policies.extend(policies_this_game)
        
        # Train on all collected data
        for idx in range(len(all_states)):
            # Prepare training data
            state = all_states[idx]
            policy = all_policies[idx]
            value = all_values[idx]
            
            # Convert to tensors
            board_tensor = state.encode().to(device)
            policy_tensor = torch.from_numpy(policy).float().to(device)
            value_tensor = torch.tensor([value], dtype=torch.float32, device=device)
            
            # Train multiple iterations on each state
            avg_loss = 0
            num_iterations = 5  # Number of training iterations per state
            for _ in range(num_iterations):
                loss = network.train_step(board_tensor, policy_tensor, value_tensor, learning_rate)
                avg_loss += loss / num_iterations

            if idx % 10 == 0:  # Print progress periodically
                losses.append(avg_loss)
                print(f"Episode {episode} Training step {idx}, Average Loss: {avg_loss:.4f}", flush=True)
        
        # Save model and plot loss periodically
        if (episode + 1) % 10 == 0:
            network.save_model(f"models/model_iter_{episode+1}.pt")
            plot_training_loss(losses)
        network.save_model("models/model_latest.pt")
    
    return network


def plot_training_loss(losses):
        """Plot the training loss history"""
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True)
        plt.legend()

        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)

        # Save the plot
        save_path = os.path.join('plots', 'training_loss.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Loss plot saved to: {os.path.abspath(save_path)}", flush=True)


def play_game(puct, mcts):
    """Play a game between two players or against self"""
    game = Gomoku(BOARD_SIZE)

    while not game.is_game_over():
        state, best_node = puct.best_move(game, iterations=800)
        move = state.last_move
        game.make_move(move)

        best_node, root = mcts.search(game, iterations=800)
        move = best_node.state.last_move
        game.make_move(move)

    return game.get_winner()


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
    trained_network = train_model()
    
    # Save the trained model
    trained_network.save_model("models/best_gomoku_model.pt")

    print("Training complete, play game mcts against trained puct", flush=True)
    # Play a game against the trained model
    puct = PUCTPlayer(1.0, Gomoku(BOARD_SIZE))
    mcts = MCTSPlayer(1.0)
    winner = play_game(puct, mcts)
    print(f"Game finished! Winner: {'Puct' if winner == 1 else 'MCTS' if winner == -1 else 'Draw'}", flush=True)

