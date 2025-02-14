import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from gomoku import BOARD_SIZE, BOARD_TENSOR, POLICY_PROBS, STATUS
import os

import torch.nn as nn
import torch.nn.functional as F

class GameNetwork(nn.Module):
    def __init__(self, board_size, device, learning_rate=0.0001):
        super().__init__()
        self.board_size = board_size
        self.device = device

        # Define network layers
        self.conv_input = nn.Conv2d(4, 256, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(256)

        # Dropout to prevent overfitting (20% probability)
        self.dropout = nn.Dropout(p=0.2)

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

        # Policy and value heads
        self.policy_conv = nn.Conv2d(256, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        self.value_conv = nn.Conv2d(256, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.95)

    def forward(self, x):
        """Forward pass through the network."""
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        # Initial convolution
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Apply dropout after the first convolution
        x = self.dropout(x)

        # Residual layers with dropout
        for layer in self.residual_layers:
            residual = x
            x = layer(x)
            x = self.dropout(x)  # Apply dropout after each residual block
            x += residual
            x = F.relu(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.reshape(-1, 2 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.reshape(-1, self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

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
    
    def train_step(self, state_tensor, policy_tensor, value_tensor):
        """Perform a single training step with debugging for NaN values."""
        self.train()
        
        # Forward pass
        predicted_policy, predicted_value = self(state_tensor.to(self.device))

        # Debugging: Check if policy or value is NaN
        if torch.isnan(predicted_policy).any() or torch.isnan(predicted_value).any():
            print(" WARNING: NaN detected in forward pass!")
            return float('nan')  # Prevents corrupt training updates

        # Normalize policy logits before softmax
        predicted_policy = F.softmax(predicted_policy / torch.norm(predicted_policy, p=2, dim=-1, keepdim=True), dim=-1)

        # Compute losses
        policy_loss = F.cross_entropy(predicted_policy, policy_tensor.to(self.device))
        value_loss = F.mse_loss(predicted_value.squeeze(), value_tensor.float().to(self.device))

        # Debugging: Check for NaN in loss
        if torch.isnan(policy_loss).any() or torch.isnan(value_loss).any():
            print(" WARNING: NaN detected in loss!")
            return float('nan')

        # Apply weighted loss
        policy_weight = 0.7
        value_weight = 0.3
        loss = policy_weight * policy_loss + value_weight * value_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()

        return loss.item()
