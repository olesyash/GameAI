import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from gomoku import BOARD_SIZE, BOARD_TENSOR, POLICY_PROBS, STATUS
import os


class GameNetwork(nn.Module):
    def __init__(self, board_size, device):
        super().__init__()
        self.board_size = board_size
        self.device = device
        
        # Network architecture
        self.conv_input = nn.Conv2d(3, 256, 3, padding=1)  
        self.bn_input = nn.BatchNorm2d(256)
        
        # Residual layers
        self.residual_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256)
            ) for _ in range(3)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(256, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # Value head
        self.value_conv = nn.Conv2d(256, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
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
            x: Input tensor of shape [3, board_size, board_size] or [batch_size, 3, board_size, board_size]
        Returns:
            tuple: (policy, value)
                - policy: tensor of shape [board_size * board_size] or [batch_size, board_size * board_size]
                - value: tensor of shape [1] or [batch_size, 1]
        """
        # Add batch dimension if not present
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        # Get valid moves mask from input
        valid_moves = x[:, 2:3]  # Shape: [batch_size, 1, board_size, board_size]
        
        # Initial convolution
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual layers
        for layer in self.residual_layers:
            residual = x
            x = layer(x)
            x += residual
            x = F.relu(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.reshape(-1, 2 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        
        # Mask illegal moves by setting their logits to -infinity
        policy = policy.reshape(-1, self.board_size * self.board_size)
        valid_moves = valid_moves.reshape(-1, self.board_size * self.board_size)
        policy = policy.masked_fill(valid_moves == 0, float('-inf'))
        policy = F.softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.reshape(-1, self.board_size * self.board_size)
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
        self.eval()
        board_tensor = state.encode().to(self.device)
        
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
    
    def train_step(self, state_tensor, policy_tensor, value_tensor, learning_rate):
        """Perform a single training step."""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        
        # Forward pass
        predicted_policy, predicted_value = self(state_tensor.to(self.device))
        
        # Calculate losses
        policy_loss = F.cross_entropy(predicted_policy, policy_tensor.to(self.device))
        value_loss = F.mse_loss(predicted_value.squeeze(), value_tensor.float().to(self.device))
        loss = policy_loss + value_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
