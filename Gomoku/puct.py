import math
from copy import deepcopy
import random
import matplotlib.pyplot as plt
import numpy as np

from nn import GameNetwork
from fake_network import FakeNetwork
import torch
import torch.nn.functional as F
from gomoku import Gomoku, BOARD_SIZE, BOARD_TENSOR, POLICY_PROBS, STATUS
import time
import os
BEST_MODEL_PATH = os.path.join("models", "best_checkpoint.pt")


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
            self.model = GameNetwork(board_size=game.board_size, device=self.device)
            self.model.to(self.device)
        if model_path is not None:
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
        for move in possible_moves:
            if move not in [child.state.last_move for child in node.children]:
                new_state = node.state.clone()
                new_state.make_move(move)
                # Convert (row, col) to index in the policy vector
                move_index = move[0] * node.state.board_size + move[1]
                move_prob = policy[move_index].item() if isinstance(policy, torch.Tensor) else policy[move_index]
                child_node = PUCTNode(new_state, parent=node, p=move_prob)
                node.children.append(child_node)
        return node

    def back_propagate(self, node, value):
        while node is not None:
            node.N += 1  # Increment the visit count
            node.Q = (node.N * node.Q + value) / (node.N + 1) # Update the value
            node = node.parent
            value = -value

    def choose_best_move(self, root_node):
        if not root_node.children:
            return None
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
                
                if node.Q == 0:  # Only set value if not already set
                    node.Q = value

                node = self.expand(node, curr_policy)

                # 3. Backpropagation
                self.back_propagate(node, -value)
        
        # Return move based on visit count distribution during training, or best move during evaluation
        if is_training:
            visit_counts = np.array([child.N for child in root.children])
            probs = visit_counts / visit_counts.sum()  # normalize to get probabilities
            chosen_child = np.random.choice(root.children, p=probs)
            return chosen_child.state.last_move, root
        return self.choose_best_move(root), root
