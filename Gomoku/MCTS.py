from copy import deepcopy
import random
import math


class MCTSPlayer:
    def __init__(self, exploration_weight=1):
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
                child_node = self.get_or_create_node(new_state, parent=node)
                node.children.append(child_node)
                return child_node
        raise Exception("No moves to expand")

    def simulate(self, node):
        """Simulation phase: Play a random game until a terminal state is reached."""
        current_state = node.state.clone()  # Clone the state for simulation
        while not current_state.is_game_over():
            move = random.choice(current_state.legal_moves())
            current_state.make_move(move)

        # non relative to current player !!! AI is always -1!!!
        # For AI to win we need -1 in the status
        if current_state.status == -1:
            return 1
        elif current_state.status == 1:
            return -1
        else:
            return 0

    def backpropagate(self, node, reward):
        """Backpropagation phase: Update the node values and visits up the tree."""
        while node is not None:
            node.visits += 1
            node.value += reward
            reward = -reward
            node = node.parent


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # The game state
        self.parent = parent  # Parent node
        self.children = []  # Child nodes
        self.visits = 0  # Number of times this node was visited
        self.value = 0  # Total reward accumulated for this node

    def is_fully_expanded(self):
        """Checks if all possible moves have been expanded."""
        return len(self.children) == len(self.state.legal_moves())

    def get_value(self, exploration_weight):
        if self.visits == 0:
            # we prioritize nodes that are not explored
            return 0 if exploration_weight == 0 else float('inf')
        else:
            return self.value / self.visits + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, exploration_weight=1.4):
        """Selects the best child node based on UCT (Upper Confidence Bound for Trees)."""
        max_value = max(self.children, key=lambda n: n.get_value(exploration_weight)).get_value(exploration_weight)
        # select nodes with the highest UCT value
        max_nodes = [n for n in self.children if n.get_value(exploration_weight) == max_value]
        # randomly select on to expand upon
        best = random.choice(max_nodes)
        return best
