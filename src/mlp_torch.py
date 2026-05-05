"""PyTorch version of the same small MLP.

Keeping this close to the NumPy version makes it easier to compare what PyTorch
automates for us: gradients, parameter updates, and tensor operations.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TorchMLP(nn.Module):
    """Simple MLP for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_torch_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dim: int = 16,
    epochs: int = 1000,
    lr: float = 0.03,
    print_every: int = 250,
    seed: int = 42,
) -> TorchMLP:
    """Train and evaluate a PyTorch MLP."""
    torch.manual_seed(seed)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    model = TorchMLP(input_dim=X_train.shape[1], hidden_dim=hidden_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)

        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test_t)
                test_classes = (test_preds >= 0.5).float()
                test_acc = (test_classes == y_test_t).float().mean().item()

            print(f"[PyTorch] epoch={epoch:4d} loss={loss.item():.4f} test_acc={test_acc:.3f}")

    return model
