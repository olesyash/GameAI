import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from gomoku import BOARD_SIZE, BOARD_TENSOR, POLICY_PROBS, STATUS
import os

class GameNetwork(nn.Module):
    def __init__(self, board_size, device):
        super(GameNetwork, self).__init__()
        self.board_size = board_size
        self.device = device
        
        # Initial convolution layer
        self.conv_input = nn.Conv2d(2, 256, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(256)
        
        # Residual layers
        self.num_residual = 10
        self.residual_layers = nn.ModuleList([
            nn.ModuleDict({
                'conv1': nn.Conv2d(256, 256, kernel_size=3, padding=1),
                'bn1': nn.BatchNorm2d(256),
                'conv2': nn.Conv2d(256, 256, kernel_size=3, padding=1),
                'bn2': nn.BatchNorm2d(256)
            }) for _ in range(self.num_residual)
        ])
        
        # Policy head - predicting move probabilities
        self.policy_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_size * board_size, board_size * board_size)
        
        # Value head - predicting game outcome
        self.value_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # He initialization for the linear layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape [2, board_size, board_size] or [batch_size, 2, board_size, board_size]
        Returns:
            tuple: (policy, value)
                - policy: tensor of shape [board_size * board_size] or [batch_size, board_size * board_size]
                - value: tensor of shape [1] or [batch_size, 1]
        """
        x = x.to(self.device)
        
        # Add batch dimension if input is 3D
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        # Initial convolution block
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for layer in self.residual_layers:
            identity = x
            out = F.relu(layer['bn1'](layer['conv1'](x)))
            out = layer['bn2'](layer['conv2'](out))
            x = F.relu(out + identity)  # Skip connection
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(x.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(x.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        # Remove batch dimension if it was added
        if len(x.shape) == 4 and x.size(0) == 1:
            policy = policy.squeeze(0)
            value = value.squeeze(0)
        
        return policy, value

    def predict(self, state):
        """Get policy and value predictions for a game state
        
        Args:
            state: Current game state
            
        Returns:
            tuple: (policy, value)
                - policy: numpy array of shape [board_size * board_size]
                - value: float value prediction (-1 to 1)
        """
        # Prepare state for neural network
        board_tensor = state.encode().to(self.device)
        
        # Add batch dimension if not present
        if len(board_tensor.shape) == 3:  # [2, board_size, board_size]
            board_tensor = board_tensor.unsqueeze(0)  # Add batch dimension
            
        # Get policy and value predictions
        with torch.no_grad():
            policy, value = self.forward(board_tensor)
            # Ensure policy is 1D array of correct size
            if isinstance(policy, torch.Tensor):
                policy = policy.cpu().numpy()
            if len(policy.shape) == 2:
                policy = policy[0]  # Take first item from batch
            # Ensure value is a scalar
            if isinstance(value, torch.Tensor):
                value = value.cpu().item()
            if hasattr(value, 'shape') and len(value.shape) > 0:
                value = value[0]
        
        return policy, value

    def save_model(self, path='models/gomoku_model.pt'):
        """Save model state to file
        
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
            'model_version': '2.0'  # Updated version for CNN architecture
        }
        torch.save(state, path)
    
    def load_model(self, path='models/gomoku_model.pt'):
        """Load model state from file
        
        Args:
            path: Path to load the model from
        """
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            if state['board_size'] != self.board_size:
                raise ValueError(f"Model board size ({state['board_size']}) does not match current board size ({self.board_size})")
            self.load_state_dict(state['state_dict'])
            print(f"Loaded model version {state.get('model_version', '1.0')} from {path}")
        else:
            print(f"No saved model found at {path}")