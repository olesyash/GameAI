from copy import deepcopy
import random
import matplotlib.pyplot as plt
import numpy as np

from nn import GameNetwork
import torch
import torch.nn.functional as F
from gomoku import Gomoku, BOARD_SIZE, BOARD_TENSOR, POLICY_PROBS, STATUS
import time


class PUCTNode:
    def __init__(self,state, parent=None, q=0, p=0):
        self.Q = q
        self.N = 0
        self.P = p
        self.children = []
        self.parent = parent
        self.state = state
        self.acting_player = 0 - state.next_player  # Current player

    def get_puct(self, exploration_weight):
        """Get the PUCT value of the node."""
        puct = self.Q + exploration_weight * self.P * (self.parent.N ** 0.5) / (1 + self.N)
        return puct

    def best_child(self, exploration_weight):
        """Select the child node with the highest PUCT value."""
        max_puct = float('-inf')
        for child in self.children:
            puct = child.get_puct(exploration_weight)
            if puct > max_puct:
                max_puct = puct
                best_child = child
        return best_child
    
    def is_fully_expanded(self):
        """Checks if all possible moves have been expanded."""
        return len(self.children) == len(self.state.legal_moves())


class PUCTPlayer:
    def __init__(self, exploration_weight, game):
        self.exploration_weight = exploration_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GameNetwork(board_size=game.board_size, device=self.device)
        self.value_weight = 0.5  # λ in loss function
        # Set device
        self.model.to(self.device)
        self.model.load_model('models/best_gomoku_model.pt')

    def select(self, node):
        """Selection phase: Navigate the tree using UCT until reaching a leaf node."""
        current = node
        while not current.state.is_game_over():
            if not current.children:  # If node has no children yet
                return current
            if not current.is_fully_expanded():
                return current
            current = current.best_child(self.exploration_weight)
        return current
    
    def expand(self, node, policy):
        """Expansion phase: Add a new child node by exploring an unvisited move."""
        if node.state.is_game_over():
            return node
        possible_moves = node.state.legal_moves()
        if not possible_moves:  # No legal moves available
            return node

        expanded_moves = {child.state.last_move for child in node.children}
        unexpanded_moves = [move for move in possible_moves if move not in expanded_moves]
        
        if not unexpanded_moves:  # All moves are expanded
            return node.best_child(self.exploration_weight)
            
        # Choose a random unexpanded move
        move = random.choice(unexpanded_moves)
        child_position = node.state.clone()
        child_position.make_move(move)
      
        # Convert (row, col) to index in the policy vector
        move_index = move[0] * node.state.board_size + move[1]
        # Get probability for this move from policy vector
        child_node = PUCTNode(child_position, parent=node, p=policy[move_index])
        node.children.append(child_node)
        return child_node
    
    def back_propagate(self, node, value):
        while node is not None:
            node.N += 1 # Increment the visit count
            node.Q += (1/node.N) * (value - node.Q) # Update the value
            node = node.parent
            value = -value

    def choose_best_move(self, root_node):
        best_child = max(root_node.children, key=lambda child: child.N)
        return best_child.state.last_move

    def best_move(self, initial_state, iterations):
        root = PUCTNode(initial_state)
        
        for _ in range(iterations):
            # 1. Selection: Traverse the tree using UCT until reaching a leaf node.
            node = self.select(root)
            # 2. Expansion: Add a new child node by exploring an unvisited move.
            if not node.state.is_game_over():
                policy, value = self.model.predict(node.state)  
                if node.Q == 0:
                    node.Q = value
                node = self.expand(node, policy)

                if root.acting_player != value:
                    value = -value

                self.back_propagate(node, value)

        # Return the best child node (without exploration weight)
        return self.choose_best_move(root)

    def train_step(self, game, optimizer):
        """Train the neural network using MCTS visit counts and game outcome
        
        Args:
            game: Current game state
            optimizer: PyTorch optimizer
        
        Returns:
            tuple: (total_loss, policy_loss, value_loss)
        """
        encoded_game = game.encode()
        board_tensor = encoded_game[BOARD_TENSOR].to(self.device)
        target_policy = encoded_game[POLICY_PROBS].to(self.device)
        target_value = encoded_game[STATUS].to(self.device)
        
        # Forward pass
        optimizer.zero_grad()
        policy, value = self.model(board_tensor)
        
        # Calculate losses
        policy_loss = -torch.sum(target_policy * torch.log(policy.view(-1) + 1e-8))  # Cross entropy
        value_loss = F.mse_loss(value, target_value)  # MSE loss
        
        # L = Policy Loss + λ · Value Loss,
        total_loss = policy_loss + self.value_weight * value_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item(), policy_loss.item(), value_loss.item()

    def plot_training_loss(self, losses, save_path='plots/training_loss.png'):
        """Plot and save the training loss graph."""
        # Ensure the plots directory exists
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='Training Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Higher DPI for better quality
        plt.close()
        
        print(f"Loss plot saved to: {os.path.abspath(save_path)}")

    def train(self, num_episodes=1000, batch_size=32, learning_rate=0.001):
        """Train the model through self-play
        
        Args:
            num_episodes: Number of games to play
            batch_size: Number of game states to train on at once
            learning_rate: Learning rate for optimizer
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        best_loss = float('inf')
        no_improvement_count = 0
        patience = 200  # Number of episodes to wait before early stopping
        
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
        print(f"Using device: {self.device}")
        
        for episode in range(num_episodes):
            epoch_start_time = time.time()
            # Initialize new game
            game = Gomoku(board_size=BOARD_SIZE)
            game_states = []
            total_loss = 0
            num_batches = 0
            
            # Play game until terminal state
            while not game.is_game_over():
                # Store current game state
                game_states.append(game.clone())
                
                # Get move from MCTS
                move = self.best_move(game, 100)
                game.make_move(move)
                
                # Train on a batch of states if we have enough
                if len(game_states) >= batch_size:
                    batch_loss = 0
                    for state in game_states[-batch_size:]:
                        loss, _, _ = self.train_step(state, optimizer)
                        batch_loss += loss
                    avg_batch_loss = batch_loss / batch_size
                    total_loss += avg_batch_loss
                    num_batches += 1
            
            # Train on remaining states at end of game
            if game_states:
                remaining = len(game_states) % batch_size
                if remaining > 0:
                    batch_loss = 0
                    for state in game_states[-remaining:]:
                        loss, _, _ = self.train_step(state, optimizer)
                        batch_loss += loss
                    avg_batch_loss = batch_loss / remaining
                    total_loss += avg_batch_loss
                    num_batches += 1
            
            # Calculate average loss for this episode
            avg_episode_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            
            # Log progress
            if episode % 10 == 0:
                print(f"Episode {episode}, Average Loss: {avg_episode_loss:.4f}")
            
            # Save best model if we have a new best loss
            if avg_episode_loss < best_loss:
                best_loss = avg_episode_loss
                self.model.save_model('models/best_gomoku_model.pt')
                print(f"Episode {episode}: New best model saved with loss {best_loss:.4f}")
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            print(f"Episode {episode + 1}/{num_episodes}, Loss: {total_loss:.4f}, Time: {epoch_duration:.2f}s")
            
            # Regular checkpoint save
            if episode % 100 == 0:
                self.model.save_model(f'models/gomoku_model_checkpoint_{episode}.pt')
            
            # Plot loss every 10 episodes
            if episode % 10 == 0:
                self.plot_training_loss([avg_episode_loss for _ in range(episode+1)])

        print("Training completed!")
        print(f"Best model saved with loss: {best_loss:.4f}")
        
        # Load the best model for future use
        self.model.load_model('models/best_gomoku_model.pt')

    def play_game(self, opponent=None):
        """Play a single game against an opponent or self
        
        Args:
            opponent: Optional opponent player (if None, plays against self)
            
        Returns:
            game: The completed game
            winner: The winning player (1 or -1) or 0 for draw
        """
        game = Gomoku(board_size=BOARD_SIZE)
        
        while not game.is_game_over():
            if game.next_player == 1 or opponent is None:
                move = self.best_move(game, 100)
            else:
                move = opponent.best_move(game, 100)
            game.make_move(move)
        
        return game, game.get_winner()