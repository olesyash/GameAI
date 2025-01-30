from copy import deepcopy

class PUCTNode:
    def __init__(self):
        self.Q = None
        self.N = None
        self.P = None
        self.children = []
        self.parent = None
        self.state = None

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
class PUCTPlayer:
    def __init__(self, exploration_weight):
        self.exploration_weight = exploration_weight

    def select(self, node):
        """Selection phase: Navigate the tree using PUCT."""
        state = deepcopy(node.state)
        while not node.state.is_game_over() and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
        return node, state

