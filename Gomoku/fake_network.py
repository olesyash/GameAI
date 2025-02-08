import numpy as np


class FakeNetwork:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.board_size = input_shape[0]

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

