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

    def train_step(self, states, root_nodes, optimizer, batch_size=32):
        """Train the neural network using batches of states and MCTS results
        
        Args:
            states: List of game states
            root_nodes: List of root nodes from MCTS search
            optimizer: PyTorch optimizer
            batch_size: Size of training batches
            
        Returns:
            tuple: (total_loss, policy_loss, value_loss)
        """
        # Convert states to batch tensor
        state_tensors = torch.stack([state.encode() for state in states])
        
        # Create target policy tensors
        board_size = states[0].board_size
        target_policies = torch.zeros(len(states), board_size * board_size)
        
        # Get winners for value targets
        winners = torch.tensor([state.get_winner() for state in states], device=self.device)
        current_players = torch.tensor([state.get_current_player() for state in states], device=self.device)
        target_values = (winners == current_players).float() * 2 - 1
        
        # Convert MCTS visit counts to policy targets
        for i, (state, root_node) in enumerate(zip(states, root_nodes)):
            total_visits = sum(child.N for child in root_node.children)
            for child in root_node.children:
                move = child.state.last_move
                move_idx = move[0] * board_size + move[1]
                target_policies[i, move_idx] = child.N / total_visits
        
        # Move tensors to device
        state_tensors = state_tensors.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device)
        
        # Calculate loss in batches
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = (len(states) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(states))
            
            # Get batch
            batch_states = state_tensors[start_idx:end_idx]
            batch_policies = target_policies[start_idx:end_idx]
            batch_values = target_values[start_idx:end_idx]
            
            # Forward pass
            optimizer.zero_grad()
            policies, values = self.model(batch_states)
            
            # Calculate losses
            policy_loss = -torch.mean(torch.sum(batch_policies * torch.log(policies + 1e-8), dim=1))
            value_loss = F.mse_loss(values.squeeze(), batch_values)
            
            # Add L2 regularization
            l2_lambda = 1e-4
            l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
            loss = policy_loss + value_loss + l2_lambda * l2_norm
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * (end_idx - start_idx)
            total_policy_loss += policy_loss.item() * (end_idx - start_idx)
            total_value_loss += value_loss.item() * (end_idx - start_idx)
        
        # Return average losses
        avg_total_loss = total_loss / len(states)
        avg_policy_loss = total_policy_loss / len(states)
        avg_value_loss = total_value_loss / len(states)
        
        return avg_total_loss, avg_policy_loss, avg_value_loss

    def train(self, num_games=1000, learning_rate=0.001, batch_size=32):
        """Train the neural network through self-play."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Initialize buffers for final states
        final_states = []
        final_nodes = []
        
        best_win_rate = 0.0
        no_improvement_count = 0
        patience = 50
        loss_history = []
        
        print(f"Starting training for {num_games} games...")
        print(f"Using device: {self.device}")
        
        for game_idx in range(num_games):
            epoch_start_time = time.time()
            game = Gomoku(self.model.board_size)
            last_root_node = None
            
            # Play a complete game
            while not game.is_game_over():
                state, root_node = self.best_move(game, 400, is_training=True)
                last_root_node = root_node
                move = self.choose_best_move(root_node).last_move
                game.make_move(move)
            
            # Store final state if game has a winner
            winner = game.get_winner()
            if winner is not None and last_root_node is not None:
                final_states.append(game.clone())
                final_nodes.append(last_root_node)
                
                # Train when we have enough states for a batch
                if len(final_states) >= batch_size:
                    loss, policy_loss, value_loss = self.train_step(
                        final_states, final_nodes, optimizer, batch_size)
                    loss_history.append(loss)
                    scheduler.step(loss)
                    
                    # Clear buffers after training
                    final_states = []
                    final_nodes = []
                    
                    # Evaluate every 20 games
                    if game_idx % 20 == 0:
                        self.model.eval()
                        win_rate = self.evaluate_model(num_games=10)
                        self.model.train()
                        
                        print(f"Game {game_idx}: Loss = {loss:.4f}, Win Rate = {win_rate:.2f}")
                        
                        # Save if win rate improves
                        if win_rate > best_win_rate:
                            best_win_rate = win_rate
                            self.model.save_model('models/best_gomoku_model.pt')
                            print(f"New best model saved with win rate {best_win_rate:.2f}")
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1
                    
                    if no_improvement_count >= patience:
                        print(f"Early stopping triggered after {game_idx} games")
                        break
            
            # Regular checkpoint save
            if game_idx % 100 == 0:
                self.model.save_model(f'models/gomoku_model_checkpoint_{game_idx}.pt')
            
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            print(f"Game {game_idx} completed in {epoch_duration:.2f} seconds")
        
        # Train on remaining states if any
        if final_states:
            loss, policy_loss, value_loss = self.train_step(
                final_states, final_nodes, optimizer, len(final_states))
        
        print("Training completed!")
        print(f"Best model saved with win rate: {best_win_rate:.2f}")
        
        # Load the best model for future use
        self.model.load_model('models/best_gomoku_model.pt')
        self.model.eval()

    def evaluate_model(self, num_games=10):
        """Evaluate current model against the previous best model.
        
        Returns:
            float: Win rate against previous best model
        """
        # Load previous best model
        previous_best = GameNetwork(self.model.board_size, self.device).to(self.device)
        try:
            previous_best.load_model('models/best_gomoku_model.pt')
        except:
            # If no previous best exists, return 1.0 (automatic win)
            return 1.0
            
        previous_best.eval()
        opponent = PUCTPlayer(1.0, Gomoku(self.model.board_size))
        opponent.model = previous_best
        
        wins = 0
        for game_idx in range(num_games):
            # Alternate playing black and white
            if game_idx % 2 == 0:
                winner = self.play_evaluation_game(opponent)
                if winner == 1:  # Current model wins as black
                    wins += 1
            else:
                winner = opponent.play_evaluation_game(self)
                if winner == -1:  # Current model wins as white
                    wins += 1
                    
        return wins / num_games

    def play_evaluation_game(self, opponent):
        """Play a single evaluation game against an opponent.
        
        Returns:
            int: Winner of the game (1 or -1)
        """
        game = Gomoku(self.model.board_size)
        
        while not game.is_game_over():
            if game.next_player == 1:
                state, _ = self.best_move(game, 400, is_training=False)
                move = state.last_move
            else:
                state, _ = opponent.best_move(game, 400, is_training=False)
                move = state.last_move
            game.make_move(move)
            
        return game.get_winner()

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
        game = Gomoku(self.model.board_size)
        
        while not game.is_game_over():
            if game.next_player == 1 or opponent is None:
                state, _ = self.best_move(game, 100)
                move = state.last_move
            else:
                state, _ = opponent.best_move(game, 100)
                move = state.last_move
            game.make_move(move)
        
        return game, game.get_winner()