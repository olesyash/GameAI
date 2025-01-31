import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class GameNetwork(nn.Module):
    # def __init__(self,board_size):
    #     super(GameNetwork, self).__init__()
    #     self.board_size = board_size
    #     pass
    #
    #     self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
    #     self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    #     self.conv3 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
    #
    #     #policy head
    #     self.policy_conv = nn.Conv2d(256, 32, kernel_size=1)
    #     self.policy_fc = nn.Linear(32*board_size*board_size, board_size*board_size)
    #
    #     #value head
    #     self.value_conv = nn.Conv2d(256, 32, kernel_size=1)
    #     self.value_fc1 = nn.Linear(32*board_size*board_size, 256)
    #     self.value_fc2 = nn.Linear(256, 1)
    #
    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = F.relu(self.conv2(x))
    #     x = F.relu(self.conv3(x))
    #
    #     #policy head
    #     policy = self.policy_conv(x)
    #     policy = policy.view(-1, 32*self.board_size*self.board_size)
    #     policy = self.policy_fc(policy)
    #     policy = F.log_softmax(policy, dim=1)
    #
    #     #value head
    #     value = self.value_conv(x)
    #     value = value.view(-1, 32*self.board_size*self.board_size)
    #     value = F.relu(self.value_fc1(value))
    #     value = torch.tanh(self.value_fc2(value))
    #
    #     return value, policy

    def __init__(self, board_size):
        super(GameNetwork, self).__init__()
        self.board_size = board_size
        input_size = board_size * board_size  # Flattened board

        # Shared layers
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)

        # Policy head
        self.policy_fc1 = nn.Linear(256, 256)
        self.policy_out = nn.Linear(256, board_size * board_size)

        # Value head
        self.value_fc1 = nn.Linear(256, 128)
        self.value_out = nn.Linear(128, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, self.board_size * self.board_size)

        # Ensure input tensor is on the same device as the model
        x = x.to(next(self.parameters()).device)

        # Shared layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        # Policy head
        policy = F.relu(self.policy_fc1(x))
        policy = self.dropout(policy)
        policy = self.policy_out(policy)
        policy = F.softmax(policy, dim=1)

        # Value head
        value = F.relu(self.value_fc1(x))
        value = self.dropout(value)
        value = self.value_out(value)
        value = torch.tanh(value)  # Ensure value is between -1 and 1

        return policy, value.view(-1, 1)  # Ensure value has shape [batch_size, 1]

    def predict(self, state):
        """Get policy and value predictions for a game state
        
        Args:
            state: Current game state
            
        Returns:
            tuple: (policy, value)
                - policy: numpy array of move probabilities
                - value: float value prediction (-1 to 1)
        """
        # Prepare state for neural network
        board_tensor = torch.FloatTensor(state.board).view(1, 1, self.board_size, self.board_size)
        board_tensor = board_tensor.to(next(self.parameters()).device)
        
        # Get policy and value predictions
        with torch.no_grad():
            policy, value = self.forward(board_tensor)
            # Move tensors to CPU before converting to numpy
            policy = policy.cpu().view(-1).numpy()
            value = value.cpu().item()

        return policy, value  # Return in order expected by PUCT

    def save_model(self, path='models/gomoku_model.pt'):
        """
        Save model state to file
        
        Args:
            path: Path to save the model to
        """
        # Create models directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        state = {
            'board_size': self.board_size,
            'state_dict': self.state_dict(),
            'model_version': '1.0'  # Useful for future compatibility
        }
        torch.save(state, path)
        print(f"Model saved to {path}")

    def load_model(self, path='models/gomoku_model.pt'):
        """
        Load model state from file
        
        Args:
            path: Path to load the model from
            
        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        try:
            # Load model state
            state = torch.load(path)
            
            # Verify board size matches
            if state['board_size'] != self.board_size:
                print(f"Warning: Saved model board size ({state['board_size']}) "
                      f"doesn't match current board size ({self.board_size})")
                return False
            
            # Load state dict
            self.load_state_dict(state['state_dict'])
            print(f"Model loaded from {path}")
            return True
            
        except FileNotFoundError:
            print(f"No saved model found at {path}")
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False