from copy import deepcopy
import random
import math
from gomoku import BLACK_WIN, WHITE_WIN, BLACK, WHITE, ONGOING
import numpy as np


class MCTSPlayer:
    def __init__(self, exploration_weight=2):
        self.exploration_weight = exploration_weight

    def search(self, initial_state, iterations=1000):
        """Performs MCTS to find the best move."""
        root = MCTSNode(initial_state)

        for _ in range(iterations):
            # 1. Selection: Traverse the tree using UCT until reaching a leaf node.
            node = self.select(root)

            # 2. Expansion: Add a new child node by exploring an unvisited move.
            if not node.state.is_game_over():
                node = self.expand(node)

            # 3. Simulation: Simulate a random game from the new node.
            reward = self.simulate(node)

            # 4. Backpropagation: Update the value and visit counts up the tree.
            self.backpropagate(node, reward)

        # Print tree before returning best move
        print("\nFinal Tree Structure:")
        self.print_tree(root)
        
        # Return the best child node (without exploration weight).
        return root.best_child(exploration_weight=0)
        
    def select(self, node):
        """Selection phase: Navigate the tree using UCT."""
        while not node.state.is_game_over() and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
        return node

    def expand(self, node):
        """Expansion phase: Add a new child node for an unvisited move."""
        possible_moves = node.state.legal_moves()
        for move in possible_moves:
            if move not in [child.state.last_move for child in node.children]:
                new_state = node.state.clone()
                new_state.make_move(move)
                child_node = MCTSNode(new_state, parent=node)
                node.children.append(child_node)
                return child_node
        raise Exception("No moves to expand")

    def simulate(self, node):
        """Simulation phase: Play out the game with a mix of heuristic and random moves."""
        current_state = node.state.clone()
        
        while not current_state.is_game_over():
            moves = current_state.legal_moves()
            
            # Check for winning moves
            for move in moves:
                test_state = current_state.clone()
                test_state.make_move(move)
                if test_state.is_game_over():
                    current_state.make_move(move)
                    break  # Just break, not continue
                    
            # If no winning moves, make a weighted random move
            if not current_state.is_game_over():
                weights = []
                for move in moves:
                    x, y = move
                    # Prefer central positions
                    center_x, center_y = current_state.board_size // 2, current_state.board_size // 2
                    dist_to_center = abs(x - center_x) + abs(y - center_y)
                    weight = 1.0 / (1 + dist_to_center)
                    weights.append(weight)
                
                total = sum(weights)
                if total > 0:
                    weights = [w/total for w in weights]
                    move = random.choices(moves, weights=weights)[0]
                else:
                    move = random.choice(moves)
                current_state.make_move(move)
        
        return node.get_reward(current_state)

    def backpropagate(self, node, reward):
        """Backpropagate values through the tree"""
        while node is not None:
            node.visits += 1
            node.value += reward
            reward = -reward
            node = node.parent

    def print_tree(self, node, depth=0, max_depth=2):
        """Print the tree structure for debugging."""
        if depth > max_depth:
            return

        indent = "  " * depth
        move_str = f"Move: {node.state.last_move}" if node.state.last_move else "Root"
        
        print(f"{indent}{move_str} [Player: {node.acting_player}, "
              f"Visits: {node.visits}, Value: {node.value:.2f}]")

        # Sort children by visits for better visualization
        if node.children:
            sorted_children = sorted(node.children, 
                                  key=lambda n: n.visits,
                                  reverse=True)
            for child in sorted_children[:3]:  # Show top 3 children
                self.print_tree(child, depth + 1, max_depth)
            if len(node.children) > 3:
                print(f"{indent}  ... {len(node.children)-3} more children")


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # The game state
        self.parent = parent  # Parent node
        self.children = []  # Child nodes
        self.visits = 0  # Number of times this node was visited
        self.value = 0  # Total reward accumulated for this node
        self.acting_player = 0 - state.next_player  # Current player

    def is_fully_expanded(self):
        """Checks if all possible moves have been expanded."""
        return len(self.children) == len(self.state.legal_moves())

    def get_value(self, exploration_weight):
        """Calculate UCT value for node selection."""
        if self.visits == 0:
            return float('inf') if exploration_weight > 0 else float('-inf')
            
        if not self.parent:  # Root node
            return self.value / self.visits if self.visits > 0 else 0
            
        # Standard UCT formula
        exploitation = self.value / self.visits
        parent_visits = max(1, self.parent.visits)  # Avoid log(0)
        exploration = exploration_weight * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def best_child(self, exploration_weight=2):
        """Select node with best UCT value."""
        if not self.children:
            return self  # Return self if no children
            
        # Get all valid children
        valid_children = [c for c in self.children if c is not None]
        if not valid_children:
            return self
            
        # Get max UCT value
        max_value = float('-inf')
        best_nodes = []
        
        for child in valid_children:
            value = child.get_value(exploration_weight)
            if value > max_value:
                max_value = value
                best_nodes = [child]
            elif value == max_value:
                best_nodes.append(child)
                
        # Among nodes with equal UCT, choose one with most visits
        return max(best_nodes, key=lambda n: n.visits)

    def get_reward(self, state):
        """Returns the reward for the current game state relative to player"""
        if state.status == self.acting_player:
            return 1.0  # Win
        elif state.status == -self.acting_player:
            return -1.0  # Loss
        elif state.status == 0 and not state.legal_moves():
            return 0.0  # Draw
        else:
            return 0.0  # Game not finished
