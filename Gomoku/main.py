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
        network.load_model("models/model_latest.pt")
        print("Loaded latest model", flush=True)
    except:
        print("No existing model found, starting fresh", flush=True)
    
    # Training parameters

    

    learning_rate = 0.0001
    num_episodes = 1000
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

    
    
    
    for episode in range(num_episodes):
        start_time = time.time()
        exploration_weight = max(0.1, 1.0 / (1 + episode / 50))  # Gradual decrease 1
        
        game = Gomoku(board_size=BOARD_SIZE)
        states_this_game = []  # Store all states in this game
        policies_this_game = []  # Store MCTS policies for each state
        
        # Create PUCT player with current network
        puct = PUCTPlayer(exploration_weight, game, fake_network=False)
        # Create MCTS player
        mcts = MCTSPlayer(exploration_weight)
        
        # Play one game
        while not game.is_game_over():
            
            if(episode % 2 == 0):
                # Get move probabilities from PUCT
                state, best_node = puct.best_move(game, iterations=800)
                if state is None:
                    break 
                move = state.last_move
                game.make_move(move)
                policy = np.zeros(BOARD_SIZE * BOARD_SIZE)
                total_visits = sum(child.N for child in best_node.children)
                if total_visits > 0:
                    for child in best_node.children:
                        move = child.state.last_move
                        move_idx = move[0] * BOARD_SIZE + move[1]
                        policy[move_idx] = child.N / total_visits

                policies_this_game.append(policy)  
                # Get move from MCTS
                best_node, root = mcts.search(game, iterations=800)
                            
                # Make the move
                move = best_node.state.last_move
                game.make_move(move)
            else:
               
                # Get move from MCTS
                best_node, root = mcts.search(game, iterations=7000)
                            
                # Make the move
                move = best_node.state.last_move
                game.make_move(move)
                
                 # Get move probabilities from PUCT
                state, _ = puct.best_move(game, iterations=7000 )
                if (state is None):
                    break
                move = state.last_move
                game.make_move(move)
            
            # Calculate policy from MCTS
            policy = np.zeros(BOARD_SIZE * BOARD_SIZE)
            total_visits = sum(child.visits for child in root.children)
            if total_visits > 0:
                for child in root.children:
                    move = child.state.last_move
                    move_idx = move[0] * BOARD_SIZE + move[1]
                    policy[move_idx] = child.visits / total_visits

            policies_this_game.append(policy) 
          

            current_state = game.clone()
            states_this_game.append(current_state)
         
       
        # Get game result
        winner = game.get_winner()
        print(f"Game finished! Winner: {'Black' if winner == 1 else 'White' if winner == -1 else 'Draw'}", flush=True)
        
        # Calculate values for all states in this game
        for state in states_this_game:
            player = state.get_current_player()
            if winner == 0:
                value = 0  # Draw
            else:
                value = 1 if winner == player else -1
            all_values.append(value)
        
        # Add states and policies to our training data
        all_states.extend(states_this_game)
        all_policies.extend(policies_this_game)
        
        # Train on all collected data
        batch_size = 128  # Instead of 32 or 64

        for epoch in range(10):  # Train multiple times on collected data
            indices = np.random.permutation(len(all_states))  # Shuffle data
            for i in range(0, len(all_states), batch_size):
                batch_indices = indices[i:i + batch_size]
                
                # Create batch tensors
                state_batch = torch.stack([all_states[idx].encode().to(device) for idx in batch_indices])
                policy_batch = torch.stack([torch.from_numpy(all_policies[idx]).float().to(device) for idx in batch_indices])
                value_batch = torch.tensor([all_values[idx] for idx in batch_indices], dtype=torch.float32, device=device)

                # Perform a training step with the entire batch
                loss = network.train_step(state_batch, policy_batch, value_batch)
                
                losses.append(loss)

            print(f"Epoch {epoch+1}, Average Loss: {loss:.4f}", flush=True)

            if idx % 10 == 0:  # Print progress periodically
                losses.append(avg_loss)
                print(f"Episode {episode} Training step {idx}, Average Loss: {avg_loss:.4f}", flush=True)
        
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
