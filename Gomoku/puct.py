from copy import deepcopy
from nn import GameNetwork


class PUCTNode:
    def __init__(self,state, parent=None, q=0, p=0):
        self.Q = q
        self.N = 0
        self.P = p
        self.children = []
        self.parent = parent
        self.state = state
        self.value_weight = value_weight  # λ in loss function

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
    def __init__(self, exploration_weight):
        self.exploration_weight = exploration_weight

    def select(self, node):
        """Selection phase: Navigate the tree using PUCT."""
        state = deepcopy(node.state)
        while not node.state.is_game_over() and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
        return node, state
    
    def expand(self,node, policy):
        """Expansion phase: Add a new child node for an unvisited move."""
        possible_moves = node.state.legal_moves()

        for move in possible_moves:
            if move not in [child.state.last_move for child in node.children]:
                child_position = node.state.clone()
                child_position.make_move(move)
                # move is (row, col) so we need to convert it to a single index
                move_index = move[0] * node.state.size + move[1]
                # policy is a vector of probabilities for each move,
                # so we need to get the probability for the current move
                child_node = PUCTNode(child_position, parent=node, P=policy[move_index])
                node.children.append(child_node)
                return child_node
        raise Exception("No moves to expand")
    
    def back_propagate(self, node, state, value):
        while node is not None:
            node.N += 1 # Increment the visit count
            node.Q += (1/node.N) * (value - node.Q) # Update the value
            node = node.parent
            value = -value


    def choose_best_move(self, root_node):
        best_child = max(root_node.children, key=lambda child: child.N)
        return best_child.move

    def best_move(self,initial_state, iterations):
        root = PUCTNode(initial_state)
        gm = GameNetwork()
        for _ in range(iterations):
            # 1. Selection: Traverse the tree using UCT until reaching a leaf node.
            node, state = self.select(root)
            # 2. Expansion: Add a new child node by exploring an unvisited move.
            if not node.state.is_game_over():
                policy, value = gm.predict(node.state)
                if node.Q == 0:
                    node.Q = value
                node = self.expand(node, policy)
                self.back_propagate(node, state, value)

        # Return the best child node (without exploration weight)
        return self.choose_best_move(root)


