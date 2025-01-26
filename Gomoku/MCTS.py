from copy import deepcopy
import random
import math


class MCTS:
    def __init__(self, exploration_weight=1):
        self.exploration_weight = exploration_weight

    def search(self, initial_state, iterations=1000):
        """Performs MCTS to find the best move."""
        root = Node(initial_state)

        for _ in range(iterations):
            # 1. Selection: Traverse the tree using UCT until reaching a leaf node.
            node, state = self.select(root)

            # 2. Expansion: Add a new child node by exploring an unvisited move.
            if not node.state.is_game_over():
                node = self.expand(node)

            # 3. Simulation: Simulate a random game from the new node.
            reward = self.simulate(node)

            # 4. Backpropagation: Update the value and visit counts up the tree.
            self.backpropagate(node, state, reward)

        # Return the best child node (without exploration weight).
        return root.best_child(exploration_weight=0)

    def select(self, node):
        """Selection phase: Navigate the tree using UCT."""
        state = deepcopy(node.state)
        while not node.state.is_game_over() and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
        return node, state

    def expand(self, node):
        """Expansion phase: Add a new child node for an unvisited move."""
        possible_moves = node.state.legal_moves()
        for move in possible_moves:
            if move not in [child.state.last_move for child in node.children]:
                new_state = node.state.clone()
                new_state.make_move(move)
                child_node = Node(new_state, parent=node)
                node.children.append(child_node)
                return child_node
        raise Exception("No moves to expand")

    def simulate(self, node):
        """Simulation phase: Play a random game until a terminal state is reached."""
        current_state = node.state.clone()  # Clone the state for simulation
        print(f"Starting simulation with player: {current_state.current_player}")
        while not current_state.is_game_over():
            move = random.choice(current_state.legal_moves())
            current_state.make_move(move)
        reward = current_state.get_reward()
        print(f"Simulation ended. Final player: {current_state.current_player}, Reward: {reward}")
        return reward

    def backpropagate(self, node, state, reward):
        """Backpropagation phase: Update the node values and visits up the tree."""
        if reward == state.current_player:
            reward = -1
        elif reward == 0:
            reward = 0
        else:
            reward = 1
        while node is not None:
            node.visits += 1
            node.value += reward
            reward = -reward
            node = node.parent


class Node:
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
        # choose the maximum visit count
        best = max(max_nodes, key=lambda n: n.visits)
        return best
