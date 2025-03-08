import math
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np

from nn import GameNetwork
from fake_network import FakeNetwork
import torch
import torch.nn.functional as F
from gomoku import Gomoku, BOARD_SIZE, BOARD_TENSOR, POLICY_PROBS, STATUS
import os
BEST_MODEL_PATH = os.path.join("models", "model_best.pt")
TIC_TAC_TOE = os.path.join("models", "model_best_tic_tac_toe_v2.pt")
N_HISTORY = 1


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
        q_value = self.Q / (self.N + 1e-8)
        u_value = exploration_weight * self.P * math.sqrt(self.parent.N) / (1 + self.N)
        return q_value + u_value

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


class PUCTPlayer:
    def __init__(self, exploration_weight, game, fake_network=False, model_path=BEST_MODEL_PATH):
        self.exploration_weight = exploration_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.value_weight = 0.5  # λ in loss function
        if fake_network:
            self.model = FakeNetwork(board_size=game.board_size)
        else:
            self.model = GameNetwork(board_size=game.board_size, device=self.device, n_history=N_HISTORY)
            self.model.to(self.device)
        if model_path is not None:
            if game.board_size == 3 and game.win_count == 3:
                self.model.load_model(TIC_TAC_TOE)
            else:
                self.model.load_model(model_path)

    def select(self, node):
        """Selection phase: Navigate the tree using UCT."""
        while not node.state.is_game_over() and node.N > 0:
            if not node.children:  # If node has no children yet
                return node
            node = node.best_child(self.exploration_weight)
        return node
    
    def expand(self, node, policy):
        """Expansion phase: Add a new child node for an unvisited move."""
        possible_moves = node.state.legal_moves()
        board_size = node.state.board_size
        
        # Create a mask for valid moves and zero out invalid moves
        valid_moves_mask = torch.zeros(board_size * board_size, device=policy.device if isinstance(policy, torch.Tensor) else None)
        for move in possible_moves:
            move_index = move[0] * board_size + move[1]
            valid_moves_mask[move_index] = 1
            
        # Apply mask and normalize probabilities
        if isinstance(policy, torch.Tensor):
            masked_policy = policy * valid_moves_mask
            # Normalize only if sum is not zero
            if masked_policy.sum() > 0:
                masked_policy = masked_policy / masked_policy.sum()
        else:
            masked_policy = policy * valid_moves_mask.cpu().numpy()
            # Normalize only if sum is not zero
            if masked_policy.sum() > 0:
                masked_policy = masked_policy / masked_policy.sum()
        
        # Create child nodes for valid moves
        for move in possible_moves:
            if move not in [child.state.last_move for child in node.children]:
                new_state = node.state.clone()
                new_state.make_move(move)
                move_index = move[0] * board_size + move[1]
                move_prob = masked_policy[move_index].item() if isinstance(masked_policy, torch.Tensor) else masked_policy[move_index]
                child_node = PUCTNode(new_state, parent=node, p=move_prob)
                node.children.append(child_node)
        return node

    def back_propagate(self, node, value):
        while node is not None:
            node.N += 1  # Increment the visit count
            node.Q = (node.N * node.Q + value) / (node.N + 1)  # Update the value
            node = node.parent
            value = -value

    def choose_best_move(self, root_node):
        if not root_node.children:
            return None
            
        # Find the maximum number of visits among all children
        max_n = max(child.N for child in root_node.children)
        
        # Get all children with the maximum number of visits
        most_visited = [child for child in root_node.children if child.N == max_n]
        
        # If there are multiple children with same N, use P as tiebreaker
        if len(most_visited) > 1:
            best_child = max(most_visited, key=lambda child: child.P)
        else:
            best_child = most_visited[0]
            
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
        
        # Get policy and value from model
        policy, value = self.model.predict(root.state)
        
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

                node = self.expand(node, curr_policy)

                # 3. Backpropagation
                self.back_propagate(node, -value)
        
        # Return move based on visit count distribution during training, or best move during evaluation
        if is_training:
            if not root.children:
                return None, root
                
            visit_counts = np.array([child.N for child in root.children])
            total_visits = visit_counts.sum()
            
            if total_visits == 0:
                # If no visits (shouldn't happen normally), choose randomly
                chosen_child = np.random.choice(root.children)
            else:
                probs = visit_counts / total_visits  # normalize to get probabilities
                chosen_child = np.random.choice(root.children, p=probs)
                
            return chosen_child.state.last_move, root
        return self.choose_best_move(root), root
