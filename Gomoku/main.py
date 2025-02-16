from puct import PUCTPlayer
import torch
import numpy as np
from nn import GameNetwork
from gomoku import BOARD_SIZE, Gomoku
import matplotlib.pyplot as plt
import os
from MCTS import MCTSPlayer
import time

MCTS_ITERATIONS = 4000
PUCT_ITERATIONS = 7000


def train_model():
    # Initialize game and network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    learning_rate = 0.001
    network = GameNetwork(BOARD_SIZE, device, learning_rate=learning_rate)
    network.to(device)  # Ensure the model is on the correct device

    try:
        network.load_model(os.path.join("models", "model_best.pt"))
        print("Loaded latest model", flush=True)
    except:
        print("No existing model found, starting fresh", flush=True)

    # Training parameters

    num_episodes = 10000
    losses = []

    # Keep track of best model
    best_win_rate = 0.0
    evaluation_frequency = 50  # Evaluate every 50 episodes

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
        # Randomly vary exploration weights for both players
        base_exploration_weight = 1.4
        exploration_weight1 = base_exploration_weight * np.random.uniform(0.8, 1.2)  # ±20% variation
        exploration_weight2 = base_exploration_weight * np.random.uniform(0.8, 1.2)  # ±20% variation

        game = Gomoku(board_size=BOARD_SIZE)
        states_this_game = []  # Store all states in this game
        policies_this_game = []  # Store MCTS policies for each state
        values_this_game = []   # Store Q-values from MCTS

        # Create MCTS players with different exploration weights
        mcts1 = MCTSPlayer(exploration_weight1)
        mcts2 = MCTSPlayer(exploration_weight2)

        # Play one game
        while not game.is_game_over():

            # Get move from MCTS
            best_node1, root1 = mcts1.search(game, iterations=MCTS_ITERATIONS)

            # Store state and Q-value from MCTS (from Black's perspective)
            current_state = game.clone()
            states_this_game.append(current_state)
            values_this_game.append(root1.value / max(1, root1.visits))  # Normalize value

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

            if game.is_game_over():
                break

            # Get move from MCTS for player 2
            best_node2, root2 = mcts2.search(game, iterations=MCTS_ITERATIONS)

            # Store state and Q-value from MCTS
            current_state = game.clone()
            states_this_game.append(current_state)
            values_this_game.append(root2.value / max(1, root2.visits))  # Normalize value

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

        # Get game result (just for logging)
        winner = game.get_winner()
        print(f"Episode {episode + 1}, Winner: {'Black' if winner == 1 else 'White' if winner == -1 else 'Draw'}", flush=True)

        # Extend training data with this game's data
        all_states.extend(states_this_game)
        all_policies.extend(policies_this_game)
        all_values.extend(values_this_game)  # Use MCTS Q-values instead of winner

        # Train once on collected data
        avg_loss = 0  # Track batch loss
        if len(all_states) == 0:  # Check for empty dataset
            print("Warning: No training data available! Skipping training.")
            return network  # Exit training early

        # Convert to numpy arrays for efficient shuffling
        states_array = np.array(all_states)
        policies_array = np.array(all_policies)
        values_array = np.array(all_values)
        
        batch_size = min(64, len(states_array))  # Ensure batch size isn't larger than dataset
        if batch_size == 0:
            print("Warning: No data to train on! Skipping training.")
            return network

        num_epochs = 3  # Number of times to shuffle and train on all data
        num_batches = max(1, len(states_array) // batch_size)  # At least 1 batch
        total_batches = num_batches * num_epochs
        
        # Train for multiple epochs, shuffling data each time
        for epoch in range(num_epochs):
            # Shuffle all arrays using the same permutation
            shuffle_indices = np.random.permutation(len(states_array))
            shuffled_states = states_array[shuffle_indices]
            shuffled_policies = policies_array[shuffle_indices]
            shuffled_values = values_array[shuffle_indices]
            
            epoch_loss = 0
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(shuffled_states))  # Ensure we don't go past array bounds
                
                # Skip empty batches
                if end_idx <= start_idx:
                    continue
                
                # Create batch tensors from shuffled arrays
                state_batch = torch.stack([shuffled_states[i].encode().to(device) for i in range(start_idx, end_idx)])
                policy_batch = torch.stack([torch.from_numpy(shuffled_policies[i]).float().to(device) for i in range(start_idx, end_idx)])
                value_batch = torch.tensor(shuffled_values[start_idx:end_idx], dtype=torch.float32, device=device)
                
                # Perform a training step with the batch
                batch_loss = network.train_step(state_batch, policy_batch, value_batch)
                epoch_loss += batch_loss
            
            if num_batches > 0:  # Only update loss if we had batches
                avg_loss += epoch_loss / total_batches
                print(f"Episode {episode+1}, Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/num_batches:.4f}", flush=True)
        
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

            # Save if it's the best model so far
            if win_rate >= best_win_rate:
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
    
    puct = PUCTPlayer(1.0, Gomoku(BOARD_SIZE), model_path=None)
    puct.model = network
    mcts = MCTSPlayer(1.0)
    
    for game_idx in range(num_games):
        if game_idx % 2 == 0:
            winner = play_game1(puct, mcts)
        else:
            winner = play_game2(puct, mcts)

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


def play_game1(puct, mcts):
    """Play a game between two players or against self"""
    game = Gomoku(BOARD_SIZE)

    while not game.is_game_over():
        state, best_node = puct.best_move(game, iterations=PUCT_ITERATIONS)
        move = state.last_move
        print(f"puct move: {move}")
        game.make_move(move)

        best_node, root = mcts.search(game, iterations=MCTS_ITERATIONS)
        move = best_node.state.last_move
        print(f"mcts move: {move}")
        game.make_move(move)

    return game.get_winner()


def play_game2(puct, mcts):
    """Play a game between two players or against self"""
    game = Gomoku(BOARD_SIZE)

    while not game.is_game_over():
        best_node, root = mcts.search(game, iterations=MCTS_ITERATIONS)
        move = best_node.state.last_move
        print(f"mcts move: {move}")
        game.make_move(move)

        state, best_node = puct.best_move(game, iterations=PUCT_ITERATIONS)
        if not state:
            break
        move = state.last_move
        print(f"puct move: {move}")
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
