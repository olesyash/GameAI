from copy import deepcopy
import random
import math

class MCTS:
    def __init__(self, exploration_weight=1):
        self.exploration_weight = exploration_weight

    def search(self, initial_state, iterations=400):
        """Performs MCTS to find the best move."""
        root = Node(initial_state)
        
        # First check for immediate winning moves or blocking moves
        moves = initial_state.legal_moves()
        for move in moves:
            # Check if we can win
            state_copy = initial_state.clone()
            state_copy.make_move(move)
            if state_copy.get_winner() is not None:
                state_copy.unmake_move()  # Clean up
                node = Node(state_copy)
                node.state.make_move(move)
                return node
                
            # Check if opponent would win
            opponent_state = initial_state.clone()
            opponent_state.current_player = -opponent_state.current_player
            opponent_state.make_move(move)
            if opponent_state.get_winner() is not None:
                opponent_state.unmake_move()  # Clean up
                node = Node(initial_state.clone())
                node.state.make_move(move)
                return node

        # If no immediate threats, do regular MCTS
        for _ in range(iterations):
            node = self.select(root)
            if not node.state.is_game_over():
                node = self.expand(node)
            reward = self.simulate(node)
            self.backpropagate(node, reward)

        return root.best_child(exploration_weight=0)

    def select(self, node):
        """Selection phase: Navigate the tree using UCT."""
        while not node.state.is_game_over() and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
        return node

    def expand(self, node):
        """Expansion phase: Add a new child node."""
        moves = node.state.legal_moves()
        for move in moves:
            if move not in [child.state.last_move for child in node.children]:
                new_state = node.state.clone()
                new_state.make_move(move)
                child_node = Node(new_state, parent=node)
                node.children.append(child_node)
                return child_node
        raise Exception("No moves to expand")

    def simulate(self, node):
        """Simulation phase: Play a random game until terminal state."""
        state = node.state.clone()
        while not state.is_game_over():
            # Try to find winning move
            moves = state.legal_moves()
            found_good_move = False
            
            # Look for winning move
            for move in moves:
                state_copy = state.clone()
                state_copy.make_move(move)
                if state_copy.get_winner() is not None:
                    state.make_move(move)
                    found_good_move = True
                    break
                    
            if not found_good_move:
                # Look for blocking move
                for move in moves:
                    state_copy = state.clone()
                    state_copy.current_player = -state_copy.current_player
                    state_copy.make_move(move)
                    if state_copy.get_winner() is not None:
                        state.current_player = -state.current_player
                        state.make_move(move)
                        state.current_player = -state.current_player
                        found_good_move = True
                        break
                        
            if not found_good_move:
                # No immediate threats, play random
                move = random.choice(moves)
                state.make_move(move)
                
        return state.get_reward()

    def backpropagate(self, node, reward):
        """Backpropagation phase: Update values up the tree."""
        while node is not None:
            node.visits += 1
            # Adjust reward based on player perspective
            node_reward = reward * node.state.current_player
            node.value += node_reward
            node = node.parent


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        """Check if all possible moves have been expanded."""
        return len(self.children) == len(self.state.legal_moves())

    def get_value(self, exploration_weight):
        """Get the UCT value of this node."""
        if self.visits == 0:
            return float('inf') if exploration_weight > 0 else float('-inf')
            
        # Get value from current player's perspective
        value = self.value / self.visits
        
        # Add exploration bonus
        if exploration_weight > 0:
            value += exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
            
        return value

    def best_child(self, exploration_weight=1.4):
        """Select the best child node."""
        if exploration_weight == 0:
            # For final selection, prioritize most promising move
            return max(self.children, key=lambda n: (n.value/n.visits if n.visits > 0 else float('-inf')))
        else:
            # For exploration, use UCT
            return max(self.children, key=lambda n: n.get_value(exploration_weight))
