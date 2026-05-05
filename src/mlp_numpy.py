"""A small two-layer MLP implemented from scratch with NumPy.

This is intentionally compact but still follows the real training steps:
forward pass, loss calculation, backpropagation, and gradient descent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .activations import relu, relu_derivative, sigmoid
except ImportError:  # Allows running files directly during quick experiments.
    from activations import relu, relu_derivative, sigmoid


@dataclass
class TrainingHistory:
    losses: list[float]
    accuracies: list[float]


class NumpyMLP:
    """Two-layer neural network for binary classification.

    Architecture:
        X -> Linear -> ReLU -> Linear -> Sigmoid
    """

    def __init__(self, input_dim: int, hidden_dim: int = 16, seed: int = 42):
        rng = np.random.default_rng(seed)

        # He-style scaling works well with ReLU and keeps the first updates stable.
        self.W1 = rng.normal(0, np.sqrt(2 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = rng.normal(0, np.sqrt(2 / hidden_dim), size=(hidden_dim, 1))
        self.b2 = np.zeros((1, 1))

    def forward(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Run a forward pass and keep intermediate values for backprop."""
        z1 = X @ self.W1 + self.b1
        h1 = relu(z1)
        z2 = h1 @ self.W2 + self.b2
        y_hat = sigmoid(z2)

        return {"X": X, "z1": z1, "h1": h1, "z2": z2, "y_hat": y_hat}

    @staticmethod
    def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary cross-entropy averaged over the batch."""
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return float(np.mean(loss))

    def backward(self, cache: dict[str, np.ndarray], y_true: np.ndarray) -> dict[str, np.ndarray]:
        """Compute gradients for all trainable parameters."""
        X = cache["X"]
        z1 = cache["z1"]
        h1 = cache["h1"]
        y_hat = cache["y_hat"]

        n = X.shape[0]

        # For sigmoid + BCE, this compact derivative is stable and common.
        dz2 = (y_hat - y_true) / n
        dW2 = h1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * relu_derivative(z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def step(self, grads: dict[str, np.ndarray], lr: float) -> None:
        """Apply one gradient descent update."""
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities."""
        return self.forward(X)["y_hat"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary class predictions."""
        return (self.predict_proba(X) >= 0.5).astype(np.float32)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        return float(np.mean(self.predict(X) == y))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1500,
        lr: float = 0.08,
        print_every: int = 250,
    ) -> TrainingHistory:
        """Train the model using full-batch gradient descent."""
        history = TrainingHistory(losses=[], accuracies=[])

        for epoch in range(1, epochs + 1):
            cache = self.forward(X)
            loss = self.binary_cross_entropy(y, cache["y_hat"])
            grads = self.backward(cache, y)
            self.step(grads, lr)

            if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
                acc = self.accuracy(X, y)
                history.losses.append(loss)
                history.accuracies.append(acc)
                print(f"[NumPy] epoch={epoch:4d} loss={loss:.4f} acc={acc:.3f}")

        return history
