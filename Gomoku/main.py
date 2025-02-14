from puct import PUCTPlayer
import torch
import numpy as np
from nn import GameNetwork
from gomoku import BOARD_SIZE, Gomoku
import matplotlib.pyplot as plt
import os
from MCTS import MCTSPlayer
import time


def train_model():
    # Initialize game and network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    network = GameNetwork(BOARD_SIZE, device)
    network.to(device)  # Ensure the model is on the correct device
   
    try:
        network.load_model(os.path.join("models", "model_best.pt"))
        print("Loaded latest model", flush=True)
    except:
        print("No existing model found, starting fresh", flush=True)
    
    # Training parameters

    

    learning_rate = 0.0001
    num_episodes = 100
    losses = []
    
    # Keep track of best model
    best_win_rate = 0.0
    evaluation_frequency = 20  # Evaluate every 50 episodes
    
    # Store all training data
    all_states = []
    all_policies = []
    all_values = []
    
    max_training_data = 5000
    all_states = all_states[-max_training_data:]
    all_policies = all_policies[-max_training_data:]
    all_values = all_values[-max_training_data:]
    # Create PUCT player with current network
    # puct = PUCTPlayer(exploration_weight, game, fake_network=False)

    for episode in range(num_episodes):
        start_time = time.time()
        exploration_weight = 1.4
        
        game = Gomoku(board_size=BOARD_SIZE)
        states_this_game = []  # Store all states in this game
        policies_this_game = []  # Store MCTS policies for each state
        

        # Create MCTS player
        mcts1 = MCTSPlayer(exploration_weight)
        mcts2 = MCTSPlayer(exploration_weight)
        
        # Play one game
        while not game.is_game_over():

            # Get move from MCTS
            best_node1, root1 = mcts1.search(game, iterations=7000)

            # Make the move
            move1 = best_node1.state.last_move
            if not game.make_move(move1):
                break

            # Calculate policy from MCTS
            policy = np.zeros(BOARD_SIZE * BOARD_SIZE)
            total_visits = sum(child.visits for child in root1.children)
            if total_visits > 0:
                for child in root1.children:
                    move = child.state.last_move
                    move_idx = move[0] * BOARD_SIZE + move[1]
                    policy[move_idx] = child.visits / total_visits

            policies_this_game.append(policy)

            current_state = game.clone()
            states_this_game.append(current_state)

            best_node2, root2 = mcts2.search(game, iterations=7000)

            # Make the move
            move2 = best_node2.state.last_move
            if not game.make_move(move2):
                break

            # Calculate policy from MCTS
            policy = np.zeros(BOARD_SIZE * BOARD_SIZE)
            total_visits = sum(child.visits for child in root2.children)
            if total_visits > 0:
                for child in root2.children:
                    move = child.state.last_move
                    move_idx = move[0] * BOARD_SIZE + move[1]
                    policy[move_idx] = child.visits / total_visits

            policies_this_game.append(policy)

            current_state = game.clone()
            states_this_game.append(current_state)
            
       
        # Get game result
        winner = game.get_winner()
        print(f"Episode {episode + 1}, Winner: {'Black' if winner == 1 else 'White' if winner == -1 else 'Draw'}", flush=True)

        all_states.extend(states_this_game)
        all_policies.extend(policies_this_game)
        all_values.extend([1 if winner == 1 else -1 if winner == -1 else 0] * len(states_this_game))

        batch_size = 64  # Instead of 32 or 64

        # Train once on collected data
        avg_loss = 0  # Track batch loss
        if len(all_states) == 0:  # Check for empty dataset
            print("Warning: No training data available! Skipping training.")
            return network  # Exit training early

        num_batches = max(1, len(all_states) // batch_size)  # Ensure denominator is not zero

        for batch_idx in range(0, len(all_states), batch_size):
            end_idx = min(batch_idx + batch_size, len(all_states))
            batch_indices = range(batch_idx, end_idx)
            
            # Create batch tensors
            state_batch = torch.stack([all_states[idx].encode().to(device) for idx in batch_indices])
            policy_batch = torch.stack([torch.from_numpy(all_policies[idx]).float().to(device) for idx in batch_indices])
            value_batch = torch.tensor([all_values[idx] for idx in batch_indices], dtype=torch.float32, device=device)

            # Perform a training step with the entire batch
            batch_loss = network.train_step(state_batch, policy_batch, value_batch)
            avg_loss += batch_loss / num_batches  # Use corrected denominator
        
        losses.append(avg_loss)  # Store loss

        print(f"Episode {episode+1}, Average Loss: {avg_loss:.4f}", flush=True)

        # Save latest model and plot loss
        network.save_model("models/model_latest.pt")
        if (episode + 1) % 10 == 0:
            plot_training_loss(losses)

        # Evaluate model periodically
        if (episode + 1) % evaluation_frequency == 0:
            print(f"\nEvaluating model after episode {episode + 1}...", flush=True)
            win_rate = evaluate_model(network)
            print('policy')
            for policy in policies_this_game:
                print(policy)
            print('states')
            for state in states_this_game:
                print(state)
                
            # Save if it's the best model so far
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                network.save_model("models/model_best.pt")
                print(f"New best model saved! Win rate: {win_rate:.2%}", flush=True)
            
            end_time = time.time()
            print(f"Episode {episode + 1} completed in {end_time - start_time:.2f} seconds", flush=True)
    
    return network


def evaluate_model(network, num_games=10):
    """Evaluate model by playing games against MCTS"""
    wins = 0
    draws = 0
    losses = 0
    
    puct = PUCTPlayer(1.0, Gomoku(BOARD_SIZE))
    puct.model = network
    mcts = MCTSPlayer(1.0)
    
    for game_idx in range(num_games):
        winner = play_game(puct, mcts)
        if winner == 1:  # PUCT wins
            wins += 1
        elif winner == -1:  # MCTS wins
            losses += 1
        else:  # Draw
            draws += 1
        print(f"Evaluation game {game_idx + 1}: {'PUCT Win' if winner == 1 else 'MCTS Win' if winner == -1 else 'Draw'}", flush=True)
    
    win_rate = (wins + 0.5 * draws) / num_games
    print(f"Evaluation complete - Win rate: {win_rate:.2%} (Wins: {wins}, Draws: {draws}, Losses: {losses})", flush=True)
    return win_rate


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
    
    # Final evaluation
    print("\nFinal model evaluation:", flush=True)
    final_win_rate = evaluate_model(trained_network, num_games=20)  # More games for final evaluation
