from elo import EloRating
from evaluate import evaluate_agents
from puct import PUCTPlayer, N_HISTORY
import torch
import numpy as np
from nn import GameNetwork
from gomoku import BOARD_SIZE, Gomoku, BLACK, WHITE
import matplotlib.pyplot as plt
import os
from MCTS import MCTSPlayer
import time
import json
from datetime import datetime

MCTS_ITERATIONS = 7000
PUCT_ITERATIONS = 7000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
evaluation_frequency = 20  # More frequent evaluation


def initialize_network(value_weight=1.0):
    # Initialize game and network
    # Training parameters
    learning_rate = 0.001
    network = GameNetwork(BOARD_SIZE, device, n_history=N_HISTORY, learning_rate=learning_rate,
                          value_weight=value_weight)
    network.to(device)  # Ensure the model is on the correct device
    try:
        network.load_model(os.path.join("models", "model_best.pt"))
        print("Loaded latest model", flush=True)
    except:
        print("No existing model found, starting fresh", flush=True)
    return network


def train_model(num_games=1, generate_game_only=False):
    network = initialize_network()

    num_episodes = num_games
    losses = []

    # Keep track of best model
    best_win_rate = 0.0

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

    games_data = []  # Store all games data

    for episode in range(num_episodes):
        start_time = time.time()
        # Randomly vary exploration weights for both players
        base_exploration_weight = 1.4
        exploration_weight1 = base_exploration_weight * np.random.uniform(0.8, 1.2)  # ±20% variation
        exploration_weight2 = base_exploration_weight * np.random.uniform(0.8, 1.2)  # ±20% variation

        game = Gomoku(board_size=BOARD_SIZE)
        states_this_game = []  # Store all states in this game
        policies_this_game = []  # Store MCTS policies for each state
        values_this_game = []  # Store Q-values from MCTS
        moves_this_game = []  # Store moves for saving

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
            moves_this_game.append((move1, game.get_current_player()))

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
            moves_this_game.append((move2, game.get_current_player()))
            # print('--------------')
            # print(game.board)
            # print('--------------')

        # Get game result (just for logging)
        # print('-----end-game--------')
        # print(game.board)
        # print('--------------')

        winner = game.get_winner()
        print(f"Episode {episode + 1}, Winner: {'Black' if winner == 1 else 'White' if winner == -1 else 'Draw'}",
              flush=True)
        print_board(game.board)
        # Save game data
        game_data = {
            "moves": moves_this_game,
            "policies": [p.tolist() for p in policies_this_game],
            "mcts_q_values": [float(v) for v in values_this_game],
            "winner": winner,
            "timestamp": datetime.now().isoformat(),
            "board_size": BOARD_SIZE
        }

        # Save game data to file
        save_game_data(game_data)

        if generate_game_only:
            continue

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
                state_batch = torch.stack(
                    [shuffled_states[i].encode(N_HISTORY).to(device) for i in range(start_idx, end_idx)])
                policy_batch = torch.stack(
                    [torch.from_numpy(shuffled_policies[i]).float().to(device) for i in range(start_idx, end_idx)])
                value_batch = torch.tensor(shuffled_values[start_idx:end_idx], dtype=torch.float32, device=device)

                # Perform a training step with the batch
                batch_loss, _, _ = network.train_step(state_batch, policy_batch, value_batch)
                epoch_loss += batch_loss

            if num_batches > 0:  # Only update loss if we had batches
                avg_loss += epoch_loss / total_batches
                print(f"Episode {episode + 1}, Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / num_batches:.4f}",
                      flush=True)

        losses.append(avg_loss)  # Store loss

        print(f"Episode {episode + 1}, Average Loss: {avg_loss:.4f}", flush=True)

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


def train_from_data_file(value_weight=1.0):
    best_win_rate = 0
    losses = []
    value_losses = []
    policy_losses = []
    network = initialize_network(value_weight=value_weight)
    all_states, all_policies, all_values = load_games("training_data_v1.json")
    start_time = time.time()

    # Train once on collected data
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

    num_epochs = 100  # Number of times to shuffle and train on all data
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
        epoch_value_loss = 0
        epoch_policy_loss = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(shuffled_states))  # Ensure we don't go past array bounds

            # Skip empty batches
            if end_idx <= start_idx:
                continue

            # Create batch tensors from shuffled arrays
            state_batch = torch.stack(
                [shuffled_states[i].encode(N_HISTORY).to(device) for i in range(start_idx, end_idx)])
            policy_batch = torch.stack(
                [torch.from_numpy(shuffled_policies[i]).float().to(device) for i in range(start_idx, end_idx)])
            value_batch = torch.tensor(shuffled_values[start_idx:end_idx], dtype=torch.float32, device=device)

            # Perform a training step with the batch
            batch_loss, batch_value_loss, batch_policy_loss = network.train_step(state_batch, policy_batch, value_batch)
            epoch_loss += batch_loss
            epoch_value_loss += batch_value_loss
            epoch_policy_loss += batch_policy_loss

        if num_batches > 0:  # Only update loss if we had batches
            avg_epoch_loss = epoch_loss / num_batches
            avg_value_loss = epoch_value_loss / num_batches
            avg_policy_loss = epoch_policy_loss / num_batches

            losses.append(avg_epoch_loss)  # Store loss for this epoch
            value_losses.append(avg_value_loss)
            policy_losses.append(avg_policy_loss)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Policy Loss: {avg_policy_loss:.4f}",
                flush=True)

        # Save latest model and plot loss
        network.save_model("models/model_latest.pt")
        if (epoch + 1) % 5 == 0:
            plot_training_loss(losses, value_losses, policy_losses)

        # Evaluate model periodically
        if (epoch + 1) % evaluation_frequency == 0:
            print(f"\nEvaluating model after epoch {epoch + 1}...", flush=True)
            win_rate = evaluate_model(network)

            # Save if it's the best model so far
            if win_rate >= best_win_rate:
                best_win_rate = win_rate
                network.save_model("models/model_best.pt")
                print(f"New best model saved! Win rate: {win_rate:.2%}", flush=True)

            end_time = time.time()
            print(f"Epoch {epoch + 1} completed in {end_time - start_time:.2f} seconds", flush=True)
            start_time = time.time()  # Reset timer for next evaluation period

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
        print(
            f"Evaluation game {game_idx + 1}: {'PUCT Win' if winner == 1 else 'MCTS Win' if winner == -1 else 'Draw'}",
            flush=True)

    win_rate = (wins + 0.5 * draws) / num_games
    print(f"Evaluation complete - Win rate: {win_rate:.2%} (Wins: {wins}, Draws: {draws}, Losses: {losses})",
          flush=True)
    return win_rate


def plot_training_loss(losses, value_losses=None, policy_losses=None):
    """Plot the training loss history"""
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Plot combined loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Combined Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Combined Training Loss Over Time')
    plt.grid(True)
    plt.legend()

    # Save the plot
    save_path = os.path.join('plots', 'training_loss.png')
    plt.savefig(save_path)
    plt.close()

    # Plot separate value and policy losses if available
    if value_losses and policy_losses:
        # Value loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(value_losses, label='Value Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Value Loss Over Time')
        plt.grid(True)
        plt.legend()
        value_path = os.path.join('plots', 'value_loss.png')
        plt.savefig(value_path)
        plt.close()

        # Policy loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(policy_losses, label='Policy Loss', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Policy Loss Over Time')
        plt.grid(True)
        plt.legend()
        policy_path = os.path.join('plots', 'policy_loss.png')
        plt.savefig(policy_path)
        plt.close()

        print(f"Loss plots saved to: {os.path.abspath('plots')}", flush=True)
    else:
        print(f"Combined loss plot saved to: {os.path.abspath(save_path)}", flush=True)


class CompactListEncoder(json.JSONEncoder):
    """Custom JSON encoder that formats lists compactly but keeps structure indented"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def encode(self, obj):
        if isinstance(obj, str):
            return f'"{obj}"'
        elif isinstance(obj, (int, float)):
            return str(obj)
        elif obj is None:
            return 'null'
        elif isinstance(obj, bool):
            return 'true' if obj else 'false'
        elif isinstance(obj, (list, tuple)):
            # Convert to list for consistency
            items = list(obj)

            # Empty list
            if not items:
                return "[]"

            # Format moves list: [[[0, 0], 1], [[1, 1], -1]]
            if isinstance(items[0], (list, tuple)) and len(items[0]) == 2 and \
                    isinstance(items[0][0], (list, tuple)) and len(items[0][0]) == 2:
                moves_str = ",".join(f"[[{move[0][0]},{move[0][1]}],{move[1]}]" for move in items)
                return f"[{moves_str}]"

            # Format policy vectors: [0.1, 0.2, 0.3]
            if all(isinstance(x, (int, float)) for x in items):
                return f"[{','.join(str(x) for x in items)}]"

            # Other lists - encode each item
            return f"[{','.join(self.encode(x) for x in items)}]"
        elif isinstance(obj, dict):
            # Handle dictionary with proper indentation
            parts = []
            for key, value in obj.items():
                encoded_value = self.encode(value)
                parts.append(f'  "{key}": {encoded_value}')
            return "{\n" + ",\n".join(parts) + "\n}"
        else:
            return super().default(obj)


def save_game_data(game_data, filename="training_data.json"):
    """Save game data to JSON file.
    
    Args:
        game_data (dict): Dictionary containing game data
        filename (str): Name of file to save to
    """
    try:
        with open(filename, 'r') as f:
            all_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_data = []

    all_data.append(game_data)

    # Generate JSON with custom formatting
    encoder = CompactListEncoder()
    json_str = "[\n  " + encoder.encode(all_data[0])
    for item in all_data[1:]:
        json_str += ",\n  " + encoder.encode(item)
    json_str += "\n]"

    with open(filename, 'w') as f:
        f.write(json_str)


def load_games(filename="training_data.json"):
    """Load games from saved JSON file and create game states for each move.
    
    Args:
        filename (str): Name of file to load from
        
    Returns:
        tuple: (all_states, all_policies, all_values) where:
            - all_states is list of board states (numpy arrays)
            - all_policies is list of policy vectors from MCTS
            - all_values is list of Q-values from MCTS
    """
    try:
        with open(filename, 'r') as f:
            all_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading games: {e}")
        return [], [], []

    all_states = []
    all_policies = []
    all_values = []

    # Track unique games by their moves
    unique_games = {}
    duplicates = 0

    for game_data in all_data:
        # Create a unique key for this game based on its moves
        moves_key = str(game_data["moves"])

        # Skip duplicate games
        if moves_key in unique_games:
            duplicates += 1
            continue

        if game_data["board_size"] != BOARD_SIZE:
            continue

        # Mark this game as processed
        unique_games[moves_key] = True

        # For each move in the game, create a state
        game = Gomoku(board_size=game_data["board_size"])
        moves_so_far = []  # Keep track of moves to restore game state

        for i, ((move_x, move_y), player) in enumerate(game_data["moves"]):
            # Save current board state
            game_copy = Gomoku(board_size=game_data["board_size"])
            game_copy.board = game.board.copy()
            game_copy.move_history = moves_so_far.copy()  # Copy current move history
            all_states.append(game_copy)

            # Get policy and Q-value for this move
            policy = np.array(game_data["policies"][i]) if game_data["policies"][i] is not None else None
            q_value = float(game_data["mcts_q_values"][i]) if game_data["mcts_q_values"][i] is not None else None

            all_policies.append(policy)
            all_values.append(q_value)

            # Update board and move history for next state
            game.board[move_x][move_y] = player
            moves_so_far.append(((move_x, move_y), player))

    print(f"Successfully loaded {len(all_states)} states from {len(unique_games)} unique games")
    if duplicates > 0:
        print(f"Skipped {duplicates} duplicate games")

    return all_states, all_policies, all_values


def play_game1(puct, mcts):
    """Play a game between two players or against self"""
    game = Gomoku(BOARD_SIZE)

    while not game.is_game_over():
        state, best_node = puct.best_move(game, iterations=PUCT_ITERATIONS)
        if not state:  # Add check for None
            break
        move1 = state.last_move
        game.make_move(move1)
        if game.is_game_over():
            break

        best_node, root = mcts.search(game, iterations=MCTS_ITERATIONS)
        if not best_node:  # Add check for None
            break
        move2 = best_node.state.last_move
        game.make_move(move2)
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
        if not best_node:  # Add check for None
            break
        move1 = best_node.state.last_move
        # print(f"mcts move: {move}")
        game.make_move(move1)
        if game.is_game_over():
            break

        state, best_node = puct.best_move(game, iterations=PUCT_ITERATIONS)
        if not state:
            break
        move2 = state.last_move
        game.make_move(move2)

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
        print(f"Episode {episode + 1}, Winner: {'Black' if winner == 1 else 'White' if winner == -1 else 'Draw'}",
              flush=True)

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
                state_batch = torch.stack(
                    [shuffled_states[i].encode(N_HISTORY).to(device) for i in range(start_idx, end_idx)])
                policy_batch = torch.stack(
                    [torch.from_numpy(shuffled_policies[i]).float().to(device) for i in range(start_idx, end_idx)])
                value_batch = torch.tensor(shuffled_values[start_idx:end_idx], dtype=torch.float32, device=device)

                # Perform a training step with the batch
                batch_loss = network.train_step(state_batch, policy_batch, value_batch)
                epoch_loss += batch_loss

            if num_batches > 0:  # Only update loss if we had batches
                avg_loss += epoch_loss / total_batches
                print(f"Episode {episode + 1}, Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / num_batches:.4f}",
                      flush=True)

        losses.append(avg_loss)  # Store loss

        print(f"Episode {episode + 1}, Average Loss: {avg_loss:.4f}", flush=True)

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


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run the diagnostic test
    # test_value_perspectives()

    # Comment out the following lines when running the test
    # Train the model using self-play with PUCT
    #trained_network = train_model_vs_itself()
    # Generate games data only
    train_model(num_games=100, generate_game_only=False)
    # trained_network = train_from_data_file(value_weight=1.0)

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
