from copy import deepcopy
import random
import matplotlib.pyplot as plt
import numpy as np

from nn import GameNetwork
import torch
import torch.nn.functional as F
from gomoku import Gomoku, BOARD_SIZE, BOARD_TENSOR, POLICY_PROBS, STATUS
import time
import os


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
        best_child = None
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

        # Use set for O(1) lookup
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
        move_prob = policy[move_index].item() if isinstance(policy, torch.Tensor) else policy[move_index]
        
        child_node = PUCTNode(child_position, parent=node, p=move_prob)
        node.children.append(child_node)
        return child_node

    def back_propagate(self, node, value):
        while node is not None:
            node.N += 1 # Increment the visit count
            node.Q += (1/node.N) * (value - node.Q)  # Update the value
            node = node.parent
            value = -value

    def choose_best_move(self, root_node):
        best_child = max(root_node.children, key=lambda child: child.N)
        return best_child.state

    def add_dirichlet_noise(self, probs, legal_moves, alpha=0.03, epsilon=0.25):
        """Add Dirichlet noise to the policy probabilities at the root node.
        
        Args:
            probs: Original policy probabilities for all moves
            legal_moves: List of legal moves
            alpha: Dirichlet noise parameter (smaller values = more concentrated)
            epsilon: Weight of noise (0 = no noise, 1 = only noise)
            
        Returns:
            1D numpy array with noisy probabilities
        """
        # Ensure probs is a 1D numpy array
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
        probs = np.asarray(probs).flatten()
        
        # Create noise array of same size as probs
        noise = np.zeros_like(probs)
        move_indices = [move[0] * self.model.board_size + move[1] for move in legal_moves]
        legal_noise = np.random.dirichlet([alpha] * len(legal_moves))
        noise[move_indices] = legal_noise
        
        # Mix noise with original probabilities using numpy operations
        noisy_probs = (1 - epsilon) * probs + epsilon * noise
        
        # Normalize only if needed
        sum_probs = np.sum(noisy_probs)
        if sum_probs > 0:
            noisy_probs /= sum_probs
            
        return noisy_probs

    def best_move(self, initial_state, iterations, is_training=False):
        curr_policy = None
        root = PUCTNode(initial_state)
        
        # Get initial policy and value for root node
        policy, value = self.model.predict(root.state)
        root.Q = value  # Set initial value for root
        
        # Add Dirichlet noise to root policy during training (AlphaZero way)
        if is_training:
            legal_moves = root.state.legal_moves()
            
            # Convert policy to numpy if it's a tensor
            if isinstance(policy, torch.Tensor):
                policy = policy.cpu().numpy()
            
            # Add Dirichlet noise to policy at root
            policy = self.add_dirichlet_noise(policy, legal_moves)
            curr_policy = torch.from_numpy(policy).float().to(self.device)

        for _ in range(iterations):
            # 1. Selection: Traverse the tree using UCT until reaching a leaf node
            node = self.select(root)
            
            # 2. Expansion and Evaluation
            if not node.state.is_game_over():
                if is_training:
                    if node != root:
                        # Get policy and value from neural network
                        curr_policy, value = self.model.predict(node.state)
                else:
                    # Get policy and value from neural network
                    curr_policy, value = self.model.predict(node.state)
                
                if node.Q == 0:  # Only set value if not already set
                    node.Q = value

                node = self.expand(node, curr_policy)
                
                # Flip value if needed based on player perspective
                if root.acting_player != node.acting_player:
                    value = -value
                    
                # 3. Backpropagation
                self.back_propagate(node, value)
        
        # Return move based on visit count distribution during training, or best move during evaluation
        if is_training:
            visit_counts = np.array([child.N for child in root.children])
            probs = visit_counts / visit_counts.sum()  # normalize to get probabilities
            chosen_child = np.random.choice(root.children, p=probs)
            return chosen_child.state.last_move, root
        return self.choose_best_move(root), root

    def train_step(self, game, root_node, optimizer):
        """Train the neural network using MCTS visit counts and game outcome
        
        Args:
            game: Current game state
            root_node: Root node from MCTS search in best_move
            optimizer: PyTorch optimizer
            
        Returns:
            tuple: (total_loss, policy_loss, value_loss)
        """
        # Create target policy from visit counts
        board_size = game.board_size
        target_policy = torch.zeros(board_size * board_size)
        total_visits = sum(child.N for child in root_node.children)

        # Convert visit counts to probabilities
        for child in root_node.children:
            move = child.state.last_move
            move_idx = move[0] * board_size + move[1]  # Convert 2D position to 1D index
            target_policy[move_idx] = child.N / total_visits
        
        # Encode current game state
        board_tensor = game.encode().to(self.device)
        target_policy = target_policy.to(self.device)
        
        # Get game outcome for value target
        winner = game.get_winner()
        target_value = torch.tensor([1.0 if winner == game.get_current_player() else -1.0], device=self.device)
        
        # Forward pass
        optimizer.zero_grad()
        policy, value = self.model(board_tensor)
        
        # Calculate losses
        policy_loss = -torch.sum(target_policy * torch.log(policy.view(-1) + 1e-8))  # Cross entropy
        value_loss = F.mse_loss(value, target_value)  # MSE loss
        
        # L = Policy Loss + λ · Value Loss
        total_loss = policy_loss + self.value_weight * value_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item(), policy_loss.item(), value_loss.item()

    def train(self, num_games=1000, learning_rate=0.001):
        """Train the neural network through self-play.
        
        Args:
            num_games: Number of self-play games to generate
            learning_rate: Initial learning rate
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_loss = float('inf')
        no_improvement_count = 0
        patience = 50  # Number of games to wait before early stopping
        loss_history = []
        
        print(f"Starting training for {num_games} games...")
        print(f"Using device: {self.device}")
        
        for game_idx in range(num_games):
            epoch_start_time = time.time()
            game = Gomoku(BOARD_SIZE)
            last_root_node = None
            
            # Play a complete game
            while not game.is_game_over():
                # Get move with Dirichlet noise and save root node
                last_root_node = self.best_move(game, 400, is_training=True)[1]
                move = self.choose_best_move(last_root_node).last_move
                game.make_move(move)
            
            # Train on the final game state if there's a winner
            winner = game.get_winner()
            if winner is not None and last_root_node is not None:
                loss, policy_loss, value_loss = self.train_step(game, last_root_node, optimizer)
                loss_history.append(loss)
                
                # Update learning rate based on loss
                scheduler.step(loss)
                
                # Early stopping check
                if loss < best_loss:
                    best_loss = loss
                    self.model.save_model('models/best_gomoku_model.pt')
                    print(f"Game {game_idx}: New best model saved with loss {best_loss:.4f}")
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                if no_improvement_count >= patience:
                    print(f"Early stopping triggered after {game_idx} games")
                    break
            
            # Regular checkpoint save
            if game_idx % 100 == 0:
                self.model.save_model(f'models/gomoku_model_checkpoint_{game_idx}.pt')
            
            # Plot loss and log progress every 10 games
            if game_idx % 10 == 0 and loss_history:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Game {game_idx}, Loss: {loss:.4f}, LR: {current_lr:.6f}")
                self.plot_training_loss(loss_history)
            
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            print(f"Game {game_idx} completed in {epoch_duration:.2f} seconds")
        
        print("Training completed!")
        print(f"Best model saved with loss: {best_loss:.4f}")
        
        # Load the best model for future use
        self.model.load_model('models/best_gomoku_model.pt')
        self.model.eval()

    def plot_training_loss(self, losses):
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
        print(f"Loss plot saved to: {os.path.abspath(save_path)}")

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
                state, _ = self.best_move(game, 100)
                move = state.last_move
            else:
                state, _ = opponent.best_move(game, 100)
                move = state.last_move
            game.make_move(move)
        
        return game, game.get_winner()