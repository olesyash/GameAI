from copy import deepcopy
import random
import math
from gomoku import BLACK_WIN, WHITE_WIN, BLACK, WHITE, ONGOING
import numpy as np


class MCTSPlayer:
    def __init__(self, exploration_weight=2):
        self.exploration_weight = exploration_weight
        self.state_dict = {}  # Dictionary to store states

    def get_state_hash(self, state):
        """Create a unique hash for a game state."""
        return hash(state.board.tobytes())

    def get_or_create_node(self, state, parent=None):
        """Get existing node from dictionary or create a new one."""
        state_hash = self.get_state_hash(state)
        if state_hash not in self.state_dict:
            self.state_dict[state_hash] = MCTSNode(state, parent)
        return self.state_dict[state_hash]

    def search(self, initial_state, iterations=1000):
        """Performs MCTS to find the best move."""
        self.state_dict.clear()  # Clear the state dictionary for new search
        root = self.get_or_create_node(initial_state)

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
        """Selection phase: Navigate the tree using UCT until reaching a leaf node."""
        current = node
        while not current.state.is_game_over():
            if not current.children:  # If node has no children yet
                return current
            if not current.is_fully_expanded():
                return current
            current = current.best_child(self.exploration_weight)
        return current

    def expand(self, node):
        """Expansion phase: Add a new child node by exploring an unvisited move."""
        if node.state.is_game_over():
            return node

        possible_moves = node.state.legal_moves()
        if not possible_moves:  # No legal moves available
            return node
            
        # Get moves not yet expanded
        expanded_moves = {child.state.last_move for child in node.children}
        unexpanded_moves = [move for move in possible_moves if move not in expanded_moves]
        
        if not unexpanded_moves:  # All moves are expanded
            return node.best_child(self.exploration_weight)
            
        # Choose a random unexpanded move
        move = random.choice(unexpanded_moves)
        new_state = node.state.clone()
        new_state.make_move(move)
        
        child_node = self.get_or_create_node(new_state, parent=node)
        node.children.append(child_node)
        return child_node

    def simulate(self, node):
        """Simulation phase: Play a random game until a terminal state is reached."""
        current_state = node.state.clone()  # Clone the state for simulation
        
        while not current_state.is_game_over():
            # First check for any winning moves
            moves = current_state.legal_moves()
            winning_move = None
            
            # First try to find winning moves
            for move in moves:
                test_state = current_state.clone()
                test_state.make_move(move)
                if test_state.status != ONGOING:
                    current_state = test_state
                    break
            
            # If no winning moves found, make a weighted random move
            if current_state.status == ONGOING:
                center_x, center_y = current_state.board_size // 2, current_state.board_size // 2
                weighted_moves = []
                for move in moves:
                    dist = abs(move[0] - center_x) + abs(move[1] - center_y)
                    weight = max(1, 5 - dist)  # Higher weight for center moves
                    weighted_moves.extend([move] * weight)
                
                move = random.choice(weighted_moves)
                current_state.make_move(move)

        # Relative to current player !!!
        return node.get_reward(current_state)

    def backpropagate(self, node, reward):
        """Backpropagate values through the tree"""
        while node is not None:
            node.visits += 1
            # Simple moving average update
            node.value += (reward - node.value) / node.visits
            reward = -reward  # Switch reward for parent
            node = node.parent

    def print_tree(self, node, depth=0, max_depth=2):
        """Print the tree structure for debugging."""
        if depth > max_depth:
            return

        indent = "  " * depth
        move_str = f"Move: {node.state.last_move}" if node.state.last_move else "Root"
        uct = 0
        if node.parent and node.visits > 0:
            exploration = self.exploration_weight * math.sqrt(math.log(node.parent.visits) / node.visits)
            exploitation = node.value / node.visits
            uct = exploitation + exploration

        print(f"{indent}{move_str} [Player: {node.acting_player}, "
              f"Visits: {node.visits}, Value: {node.value:.2f}, "
              f"UCT: {uct:.2f}]")

        # Sort children by UCT value for better visualization
        if node.children:
            sorted_children = sorted(node.children, 
                                  key=lambda n: n.value/n.visits + 
                                  self.exploration_weight * math.sqrt(math.log(node.visits)/n.visits) 
                                  if n.visits > 0 else float('inf'),
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
        if not self.state.legal_moves():
            return True
        return len(self.children) == len(self.state.legal_moves())
        
    def best_child(self, exploration_weight=2):
        """Select node with best UCT value.
        Returns:
            Node: Best child.
        """
        if self.visits == 0:
            # we prioritize nodes that are not explored
            return float('-inf') if exploration_weight == 0 else float('inf')
        best_uct = -np.inf
        best_node = None
        for child in self.children:
            uct = (child.value / child.visits) + exploration_weight * np.sqrt(np.log(self.visits) / child.visits)
            if uct > best_uct:
                best_uct = uct
                best_node = child
        # Avoid error if node has no children
        if best_node is None:
            return self
        return best_node

    def get_reward(self, state):
        """Returns the reward for the current game state relative to player"""
        if state.status == self.acting_player:
            return 1
        elif state.status == 0:  # Draw
            return 0
        else:
            return -1
