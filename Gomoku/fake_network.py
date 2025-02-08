import numpy as np


class FakeNetwork:
    def __init__(self, board_size):
        self.board_size = board_size

    def predict(self, state):
        # Create policy with zeros
        policy = np.zeros(self.board_size * self.board_size)
        
        # Get legal moves and set uniform probability
        legal_moves = state.legal_moves()
        if legal_moves:
            prob = 1.0 / len(legal_moves)
            for move in legal_moves:
                idx = move[0] * self.board_size + move[1]
                policy[idx] = prob
                
        value = 0.0  # Neutral value
        return policy, value

    def load_model(self, path):
        pass