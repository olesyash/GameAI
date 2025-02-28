from elo import EloRating
from evaluate import evaluate_agents
from puct import PUCTPlayer
import torch
import numpy as np
from nn import GameNetwork
from gomoku import BOARD_SIZE, Gomoku, BLACK, WHITE
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
    n_history = 3  # Number of historical moves to track
    network = GameNetwork(BOARD_SIZE, device, n_history=n_history, learning_rate=learning_rate)
    network.to(device)  # Ensure the model is on the correct device

    try:
        network.load_model(os.path.join("models", "model_best.pt"))
        print("Loaded latest model", flush=True)
    except:
        print("No existing model found, starting fresh", flush=True)

    # Training parameters

    num_episodes = 1000
    losses = []

    # Keep track of best model
    best_win_rate = 0.0
    evaluation_frequency = 5  # Evaluate every 50 episodes

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
            values_this_game.append(-root2.value / max(1, root2.visits))  # Normalize value

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
            print('--------------')
            print(game.board)
            print('--------------')

        # Get game result (just for logging)
        print('-----end-game--------')
        print(game.board)
        print('--------------')
 
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
            print(f"Episode {episode + 1} completed in {end_time - start_time:.2f}  seconds", flush=True)

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
            winner *= -1

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


def plot_elo_history(elo_system, save_path="plots/elo_history.png"):
    """Plot ELO rating history of models over training."""
    plt.figure(figsize=(10, 6))
    
    # Get all model versions except 'Model_Best'
    models = sorted([name for name in elo_system.history.keys() 
                    if name.startswith("Model_") and name != "Model_Best"],
                   key=lambda x: int(x.split("_")[1]))
    
    if not models:  # No models to plot yet
        return
        
    best_history = elo_system.history["Model_Best"]
    
    # Plot each model's rating
    episodes = [int(model.split("_")[1]) for model in models]
    ratings = [elo_system.history[model][-1] for model in models]  # Final rating for each model
    
    plt.plot(episodes, ratings, 'b.-', label='Current Model')
    plt.plot(episodes, best_history[-len(episodes):], 'r.-', label='Best Model')
    
    plt.title("ELO Ratings Over Training")
    plt.xlabel("Episode")
    plt.ylabel("ELO Rating")
    plt.legend()
    plt.grid(True)
    
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"ELO history plot saved to: {save_path}")


def play_game1(puct, mcts):
    """Play a game between two players or against self"""
    game = Gomoku(BOARD_SIZE)

    while not game.is_game_over():
        state, best_node = puct.best_move(game, iterations=PUCT_ITERATIONS)
        move = state.last_move
        # print(f"puct move: {move}")
        game.make_move(move)
        if game.is_game_over():
            break
        
        best_node, root = mcts.search(game, iterations=MCTS_ITERATIONS)
        move = best_node.state.last_move
        game.make_move(move)
        # print(f"mcts move: {move}")
        # print('--------------------')
        # print(game.board)
        # print('-----------------')
    
    # print('---------end-game-----------')
    # print(game.board)
    # print('-----------------')
    # print(F'thw winner {game.get_winner()}')
    return game.get_winner()


def play_game2(puct, mcts):
    """Play a game between two players or against self"""
    game = Gomoku(BOARD_SIZE)

    while not game.is_game_over():
        best_node, root = mcts.search(game, iterations=MCTS_ITERATIONS)
        move = best_node.state.last_move
        # print(f"mcts move: {move}")
        game.make_move(move)
        if game.is_game_over():
            break

        state, best_node = puct.best_move(game, iterations=PUCT_ITERATIONS)
        if not state:
            break
        move = state.last_move
        # print(f"puct move: {move}")
        # game.make_move(move)
        # print('--------------------')
        # print(game.board)
        # print('-----------------')
    
    # print('---------end-game-----------')
    # print(game.board)
    # print('-----------------')
    # print(F'thw winner {game.get_winner()}')

    return game.get_winner()

def print_board(board):
    """Print the game board in a readable format"""
    symbols = {1: 'X', -1: 'O', 0: '.'}
    for row in board:
        print(' '.join(symbols[cell] for cell in row))
    print()


def train_model_vs_itself():
    """Train the model using PUCT self-play and evaluate progress using ELO ratings."""
    # Initialize game and network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    learning_rate = 0.001
    n_history = 3  # Number of historical moves to track
    network = GameNetwork(BOARD_SIZE, device, n_history=n_history, learning_rate=learning_rate)
    network.to(device)  # Ensure the model is on the correct device

    # Initialize ELO rating system
    elo_system = EloRating()

    try:
        network.load_model(os.path.join("models", "model_best.pt"))
        print("Loaded latest model", flush=True)
    except:
        print("No existing model found, starting fresh", flush=True)

    # Training parameters
    num_episodes = 100
    evaluation_frequency = 20  # Evaluate every N episodes
    puct_iterations = 1600  # Iterations for PUCT search
    losses = []

    # Initialize replay buffer with maximum size
    max_buffer_size = 5000  # Limit buffer size to prevent old experiences from dominating
    all_states = []
    all_policies = []
    all_values = []

    game = Gomoku(board_size=BOARD_SIZE)
    puct1 = PUCTPlayer(1.4, game)
    puct1.model = network  # Set the network for player 1
    puct2 = PUCTPlayer(1.4, game)
    puct2.model = network  # Set the network for player 2
    for episode in range(num_episodes):
        episode_start_time = time.time()

        # Create PUCT players with slightly different exploration weights for diversity
        base_exploration = 1.4
        exploration1 = base_exploration * np.random.uniform(0.8, 1.2)
        exploration2 = base_exploration * np.random.uniform(0.8, 1.2)

        game = Gomoku(board_size=BOARD_SIZE)
        puct1.exploration_weight = exploration1
        puct1.game = game
        puct2.exploration_weight = exploration2
        puct2.game = game
        states_this_game = []
        policies_this_game = []
        values_this_game = []  # Store Q-values from PUCT search

        # Play one game
        while not game.is_game_over():
            current_puct = puct1 if game.next_player == BLACK else puct2

            # Get move and root node from PUCT search
            move, root = current_puct.best_move(game, puct_iterations, is_training=True)

            if move is None:
                break

            # Calculate policy from visit counts
            board_size = game.board_size
            policy = np.zeros(board_size * board_size)
            total_visits = sum(child.N for child in root.children)

            if total_visits > 0:  # Prevent division by zero
                for child in root.children:
                    if child.state.last_move:
                        move_idx = child.state.last_move[0] * board_size + child.state.last_move[1]
                        policy[move_idx] = child.N / total_visits

            # Get value estimate from root node
            value = root.Q / max(1, root.N)  # Use Q-value from PUCT search

            # Store state, policy and value
            states_this_game.append(game.clone())
            policies_this_game.append(policy)
            values_this_game.append(value)  # Store Q-value instead of final result

            # Make the move
            game.make_move(move)

        # Get game result
        winner = game.get_winner()
        print(f"Episode {episode + 1}, Winner: {'Black' if winner == 1 else 'White' if winner == -1 else 'Draw'}", flush=True)
        
        # Update training data with TD-lambda style targets
        # Blend PUCT search values with final outcome for better targets
        lambda_param = 0.7  # Weight between search value and final outcome
        for i in range(len(values_this_game)):
            # Adjust value target based on player perspective
            player_perspective = 1 if i % 2 == 0 else -1  # Alternate perspective based on player
            final_outcome = winner * player_perspective  # Adjust winner from game perspective to player perspective
            
            # Blend search value with final outcome
            values_this_game[i] = (1 - lambda_param) * values_this_game[i] + lambda_param * final_outcome
        
        # Update training data
        all_states.extend(states_this_game)
        all_policies.extend(policies_this_game)
        all_values.extend(values_this_game)
        
        # Maintain buffer size
        if len(all_states) > max_buffer_size:
            all_states = all_states[-max_buffer_size:]
            all_policies = all_policies[-max_buffer_size:]
            all_values = all_values[-max_buffer_size:]

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

        # Periodic evaluation and model saving
        if (episode + 1) % evaluation_frequency == 0:
            # Evaluate against previous best model periodically
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"Average loss: {np.mean(losses[-100:]) if losses else 'N/A'}")

            # Use evaluate.py to play games between current and previous model
            current_agent = PUCTPlayer(base_exploration, game)
            current_agent.model = network  # Use the current network directly
            prev_agent = PUCTPlayer(base_exploration, game)
  
            print("\nEvaluating current model against previous best...")
            win_rate = evaluate_agents(
                current_agent, prev_agent,
                f"Model_{episode + 1}", "Model_Best",
                num_games=20, board_size=BOARD_SIZE,
                elo_system=elo_system
            )

            # Get updated ELO ratings
            current_elo = elo_system.get_rating(f"Model_{episode + 1}")
            best_elo = elo_system.get_rating("Model_Best")

            print(f"ELO Ratings - Current: {current_elo}, Previous Best: {best_elo}")

            # Save if new model is significantly better (win rate > 55% and higher ELO)
            if win_rate > 0.55:
                elo_improvement = current_elo - best_elo
                network.save_model("models/model_best.pt")
                print(f"New best model saved! Win rate: {win_rate:.1%}, ELO improvement: {elo_improvement:.1f}")
            else:
                print(f"Model not saved. Win rate: {win_rate:.1%} (needs >55% and higher ELO to save)")

            # Update plots
            if losses:
                plot_training_loss(losses)
            plot_elo_history(elo_system)

    return network


def test_value_perspectives():
    """Test function to verify how values are interpreted in PUCT and MCTS."""
    print("\n=== Testing Value Perspectives ===")
    
    # Initialize game and players
    game = Gomoku(board_size=BOARD_SIZE)
    
    # Create MCTS player
    mcts = MCTSPlayer(exploration_weight=1.4)
    
    # Create PUCT player with network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = GameNetwork(BOARD_SIZE, device)
    network.to(device)
    try:
        network.load_model(os.path.join("models", "model_best.pt"))
        print("Loaded model successfully")
    except:
        print("No model found, using fresh network")
    
    puct = PUCTPlayer(1.4, game)
    puct.model = network
    
    # Create a simple test case with a clear advantage for Black
    # Black has 3 in a row and is about to win
    print("\nCreating a test position with advantage for Black:")
    game.board[3, 1] = BLACK  # Black
    game.board[3, 2] = BLACK  # Black
    game.board[3, 3] = BLACK  # Black
    game.next_player = BLACK  # Black to move (should be advantageous)
    
    print("\nBoard state (Black to move, Black has advantage):")
    print_board(game.board)
    
    # Test MCTS value
    print("\n--- MCTS Value Test (Black to move) ---")
    _, mcts_root = mcts.search(game.clone(), iterations=1000)
    mcts_value = mcts_root.value / max(1, mcts_root.visits)
    print(f"MCTS value (from root): {mcts_value:.4f}")
    print(f"Current player: {'Black' if game.next_player == 1 else 'White'} ({game.next_player})")
    print(f"Interpretation: {'Favorable for current player' if mcts_value > 0 else 'Unfavorable for current player'}")
    
    # Test PUCT value
    print("\n--- PUCT Value Test (Black to move) ---")
    _, puct_root = puct.best_move(game.clone(), iterations=1000, is_training=True)
    puct_value = puct_root.Q / max(1, puct_root.N)
    print(f"PUCT value (from root): {puct_value:.4f}")
    print(f"Current player: {'Black' if game.next_player == 1 else 'White'} ({game.next_player})")
    print(f"Interpretation: {'Favorable for current player' if puct_value > 0 else 'Unfavorable for current player'}")
    
    # Now switch to White's perspective
    game.next_player = WHITE  # White to move (should be disadvantageous)
    
    print("\nSame board state (White to move, White has disadvantage):")
    print_board(game.board)
    
    # Test MCTS value
    print("\n--- MCTS Value Test (White to move) ---")
    _, mcts_root = mcts.search(game.clone(), iterations=1000)
    mcts_value = mcts_root.value / max(1, mcts_root.visits)
    print(f"MCTS value (from root): {mcts_value:.4f}")
    print(f"Current player: {'Black' if game.next_player == 1 else 'White'} ({game.next_player})")
    print(f"Interpretation: {'Favorable for current player' if mcts_value > 0 else 'Unfavorable for current player'}")
    
    # Test PUCT value
    print("\n--- PUCT Value Test (White to move) ---")
    _, puct_root = puct.best_move(game.clone(), iterations=1000, is_training=True)
    puct_value = puct_root.Q / max(1, puct_root.N)
    print(f"PUCT value (from root): {puct_value:.4f}")
    print(f"Current player: {'Black' if game.next_player == 1 else 'White'} ({game.next_player})")
    print(f"Interpretation: {'Favorable for current player' if puct_value > 0 else 'Unfavorable for current player'}")
    
    # Test with a different position where White has advantage
    game = Gomoku(board_size=BOARD_SIZE)
    game.board[2, 1] = WHITE  # White
    game.board[2, 2] = WHITE  # White
    game.board[2, 3] = WHITE  # White
    game.next_player = WHITE  # White to move (should be advantageous)
    
    print("\nNew board state (White to move, White has advantage):")
    print_board(game.board)
    
    # Test MCTS value
    print("\n--- MCTS Value Test (White to move with advantage) ---")
    _, mcts_root = mcts.search(game.clone(), iterations=1000)
    mcts_value = mcts_root.value / max(1, mcts_root.visits)
    print(f"MCTS value (from root): {mcts_value:.4f}")
    print(f"Current player: {'Black' if game.next_player == 1 else 'White'} ({game.next_player})")
    print(f"Interpretation: {'Favorable for current player' if mcts_value > 0 else 'Unfavorable for current player'}")
    
    # Test PUCT value
    print("\n--- PUCT Value Test (White to move with advantage) ---")
    _, puct_root = puct.best_move(game.clone(), iterations=1000, is_training=True)
    puct_value = puct_root.Q / max(1, puct_root.N)
    print(f"PUCT value (from root): {puct_value:.4f}")
    print(f"Current player: {'Black' if game.next_player == 1 else 'White'} ({game.next_player})")
    print(f"Interpretation: {'Favorable for current player' if puct_value > 0 else 'Unfavorable for current player'}")
    
    # Now switch to Black's perspective
    game.next_player = BLACK  # Black to move (should be disadvantageous)
    
    print("\nSame board state (Black to move, Black has disadvantage):")
    print_board(game.board)
    
    # Test MCTS value
    print("\n--- MCTS Value Test (Black to move with disadvantage) ---")
    _, mcts_root = mcts.search(game.clone(), iterations=1000)
    mcts_value = mcts_root.value / max(1, mcts_root.visits)
    print(f"MCTS value (from root): {mcts_value:.4f}")
    print(f"Current player: {'Black' if game.next_player == 1 else 'White'} ({game.next_player})")
    print(f"Interpretation: {'Favorable for current player' if mcts_value > 0 else 'Unfavorable for current player'}")
    
    # Test PUCT value
    print("\n--- PUCT Value Test (Black to move with disadvantage) ---")
    _, puct_root = puct.best_move(game.clone(), iterations=1000, is_training=True)
    puct_value = puct_root.Q / max(1, puct_root.N)
    print(f"PUCT value (from root): {puct_value:.4f}")
    print(f"Current player: {'Black' if game.next_player == 1 else 'White'} ({game.next_player})")
    print(f"Interpretation: {'Favorable for current player' if puct_value > 0 else 'Unfavorable for current player'}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run the diagnostic test
    # test_value_perspectives()
    
    # Comment out the following lines when running the test
    # Train the model using self-play with PUCT
    #trained_network = train_model_vs_itself()
    trained_network = train_model()
    # Final evaluation
    # print("\nFinal model evaluation:", flush=True)
    # game = Gomoku(board_size=BOARD_SIZE)
    # Create agents for evaluation
    # final_agent = PUCTPlayer(1.4, game)
    # final_agent.model = trained_network
    # best_agent = PUCTPlayer(1.4, game)  # Best Puct as baseline
    
    # Evaluate final model against pure MCTS baseline
    # evaluate_agents(
    #     final_agent, best_agent,
    #     "Final_Model", "BestPuct_Model",
    #     num_games=50, board_size=BOARD_SIZE
    # )
