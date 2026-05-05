"""Activation functions used by the NumPy neural network.

The functions are intentionally small and explicit. I kept them separate from
the model code because it makes the math easier to test and debug.
"""

from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x)."""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU with respect to its input."""
    return (x > 0).astype(float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid activation."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid with respect to its input."""
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation."""
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of tanh with respect to its input."""
    t = np.tanh(x)
    return 1.0 - t**2


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU activation."""
    return np.where(x >= 0, x, alpha * x)


def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Derivative of Leaky ReLU."""
    return np.where(x >= 0, 1.0, alpha)


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ELU activation."""
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1.0))


def elu_derivative(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Derivative of ELU."""
    return np.where(x >= 0, 1.0, alpha * np.exp(x))
