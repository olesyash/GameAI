import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from gomoku import BOARD_SIZE, BOARD_TENSOR, POLICY_PROBS, STATUS
import os

import torch.nn as nn
import torch.nn.functional as F


"""AlphaZero Neural Network component."""
import math
from typing import NamedTuple, Tuple
import torch
from torch import nn
import torch.nn.functional as F


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
        # Policy cross-entropy loss
        policy_loss = F.cross_entropy(log_ps, target_ps, reduction='mean')

        # State value MSE loss
        value_loss = F.mse_loss(vs.squeeze(), target_vs, reduction='mean')
        # value_loss = torch.mean(torch.pow(vs - target_vs, 2))
        # policy_loss = -torch.mean(torch.sum(target_ps * log_ps, 1))

        # Apply weighting to prioritize value learning if needed
        total_loss = (self.value_weight * value_loss) + policy_loss

        return total_loss, value_loss, policy_loss

class NetworkOutputs(NamedTuple):
    pi_prob: torch.Tensor
    value: torch.Tensor


def calc_conv2d_output(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """takes a tuple of (h,w) and returns a tuple of (h,w)"""

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


def initialize_weights(net: nn.Module) -> None:
    """Initialize weights for Conv2d and Linear layers using kaming initializer."""
    assert isinstance(net, nn.Module)

    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

            if module.bias is not None:
                nn.init.zeros_(module.bias)


class ResNetBlock(nn.Module):
    """Basic redisual block."""

    def __init__(
        self,
        num_filters: int,
    ) -> None:
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = F.relu(out)
        return out


class GameNetwork(nn.Module):
    """Policy network for AlphaZero agent."""

    def __init__(
        self,
        board_size, device, num_stack=3, learning_rate=0.0001,
            value_weight=3.0,  weight_decay=0.0001) -> None:
        super().__init__()
        self.device = device
        self.board_size = board_size

        input_shape = (num_stack * 2 + 3, board_size, board_size)
        num_actions = board_size * board_size
        num_res_block = 10
        num_filters = 40
        num_fc_units = 80
        c, h, w = input_shape

        # We need to use additional padding for Gomoku to fix agent shortsighted on edge cases
        num_padding = 3

        conv_out_hw = calc_conv2d_output((h, w), 3, 1, num_padding)
        # FIX BUG, Python 3.7 has no math.prod()
        conv_out = conv_out_hw[0] * conv_out_hw[1]

        # First convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=c,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=num_padding,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(),
        )

        # Residual blocks
        res_blocks = []
        for _ in range(num_res_block):
            res_blocks.append(ResNetBlock(num_filters))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=2,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * conv_out, num_actions),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * conv_out, num_fc_units),
            nn.ReLU(),
            nn.Linear(num_fc_units, 1),
            nn.Tanh(),
        )

        initialize_weights(self)
        self.alpha_loss = AlphaLoss(value_weight=value_weight)
        self.optimizer = torch.optim.AdamW(self.parameters(),
                                           lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=50, gamma=0.95)

    def forward(self, x: torch.Tensor) -> NetworkOutputs:
        """Given raw state x, predict the raw logits probability distribution for all actions,
        and the evaluated value, all from current player's perspective."""

        conv_block_out = self.conv_block(x)
        features = self.res_blocks(conv_block_out)

        # Predict raw logits distributions wrt policy
        pi_logits = self.policy_head(features)

        # Predict evaluated value from current player's perspective.
        value = self.value_head(features)

        return pi_logits, value

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
            policy, value = self.forward(board_tensor)

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
            'model_version': '3.0'  # Updated version for CNN architecture
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

        loss, value_loss, policy_loss = self.alpha_loss(predicted_policy, predicted_value, policy_tensor, value_tensor)
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()

        return loss.item(), value_loss.item(), policy_loss.item()
