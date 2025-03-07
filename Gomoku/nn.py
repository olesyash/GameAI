import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from gomoku import BOARD_SIZE, BOARD_TENSOR, POLICY_PROBS, STATUS
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def conv3x3(in_channels, out_channels, stride=1):
    # 3x3 convolution
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    # Residual block
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = False
        if in_channels != out_channels or stride != 1:
            self.downsample = True
            self.downsample_conv = conv3x3(in_channels, out_channels, stride=stride)
            self.downsample_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample_conv(residual)
            residual = self.downsample_bn(residual)

        out += residual
        out = self.relu(out)
        return out



class AlphaLoss(nn.Module):
    """
    Loss as described in the AlphaZero paper, a weighted sum of:
    - MSE on value prediction
    - Cross-entropy on policy

    L = (z - v)² - π^T * log(p)
    where:
    z : the expected reward
    v : our prediction of the value
    π : the search probabilities
    p : probas

    The loss is then averaged over the entire batch
    """

    def __init__(self, value_weight=1.0):
        super(AlphaLoss, self).__init__()
        self.value_weight = value_weight

    def forward(self, log_ps, vs, target_ps, target_vs):
        value_loss = torch.mean(torch.pow(vs - target_vs, 2))
        policy_loss = -torch.mean(torch.sum(target_ps * log_ps, 1))
        
        # Apply weighting to prioritize value learning if needed
        total_loss = (self.value_weight * value_loss) + policy_loss
        
        return total_loss, value_loss, policy_loss


class GameNetwork(nn.Module):
    def __init__(self, board_size, device, n_history=3, learning_rate=0.001, value_weight=1.0):
        """Initialize the neural network.
        
        Args:
            board_size (int): Size of the game board
            device (torch.device): Device to run the network on
            n_history (int): Number of historical moves to include for each player
            learning_rate (float): Learning rate for optimization
            value_weight (float): Weight for the value loss component (higher = more focus on value)
        """
        super().__init__()
        self.board_size = board_size
        self.device = device
        self.n_history = n_history
        self.value_weight = value_weight
        
        # Network architecture
        # Input channels: 2 players * n_history moves + 1 current player + 2 current boards
        self.input_channels = (2 * n_history) + 3
        self.num_channels = 128
        self.num_res_blocks = 6
        
        # Input layers
        self.conv1 = nn.Conv2d(self.input_channels, self.num_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(self.num_channels, self.num_channels) for _ in range(self.num_res_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(self.num_channels, 8, 1)
        self.policy_bn = nn.BatchNorm2d(8)
        self.policy_fc = nn.Linear(8 * board_size * board_size, board_size * board_size)
        
        # Value head
        self.value_conv = nn.Conv2d(self.num_channels, 4, 1)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * board_size * board_size, 512)
        self.value_fc2 = nn.Linear(512, 1)
        
        # Loss function
        self.loss_fn = AlphaLoss(value_weight)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.001)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.to(device)

    def forward(self, x):
        """Forward pass through the network."""
        # residual block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        for block in self.res_blocks:
            out = block(out)

        # policy head
        p = self.policy_conv(out)
        p = self.policy_bn(p)
        p = F.relu(p)

        p = self.policy_fc(p.view(p.size(0), -1))
        p = F.log_softmax(p, dim=1)

        # value head
        v = self.value_conv(out)
        v = self.value_bn(v)
        v = F.relu(v)

        v = self.value_fc1(v.view(v.size(0), -1))
        v = F.relu(v)
        v = self.value_fc2(v)
        v = F.tanh(v)

        return p, v

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
        board_tensor = board_tensor.unsqueeze(0)  # Add batch dimension
        
        # Get policy and value predictions
        with torch.no_grad():
            policy, value = self(board_tensor)

            # Convert to probabilities because log_softmax returns log probabilities
            policy = torch.exp(policy)

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
            'model_version': '4.0'  # Updated version for CNN architecture
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

        loss, value_loss, policy_loss = self.loss_fn(predicted_policy, predicted_value, policy_tensor, value_tensor)
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step(loss.item())

        return loss.item(), value_loss.item(), policy_loss.item()
