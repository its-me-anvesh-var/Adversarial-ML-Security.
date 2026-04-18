#!/usr/bin/env python3
"""
============================================================
model.py — Neural Network Model Definition
Author: Anvesh Raju Vishwaraju
============================================================
"""
import torch
import torch.nn as nn


class SOCAlertNet(nn.Module):
    """
    Simple feed-forward neural network for SOC alert classification.
    Used to demonstrate adversarial ML attacks in security context.

    Architecture:
        Input(8) → FC(64) → ReLU → Dropout
                 → FC(128) → ReLU → Dropout
                 → FC(64)  → ReLU
                 → FC(2)   → Output (TP / FP)
    """

    def __init__(self, input_dim: int = 8, num_classes: int = 2,
                 dropout: float = 0.3):
        super(SOCAlertNet, self).__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3
            nn.Linear(128, 64),
            nn.ReLU(),

            # Output
            nn.Linear(64, num_classes)
        )

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SimpleSOCNet(nn.Module):
    """
    Simpler model for quick demo purposes.
    Input(8) → FC(32) → ReLU → FC(16) → ReLU → FC(2)
    """
    def __init__(self, input_dim: int = 8, num_classes: int = 2):
        super(SimpleSOCNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def load_model(path: str, model_class=SOCAlertNet,
               input_dim: int = 8) -> nn.Module:
    """Load a saved model from disk."""
    model = model_class(input_dim=input_dim)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def save_model(model: nn.Module, path: str):
    """Save model state dict to disk."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[✓] Model saved to {path}")
