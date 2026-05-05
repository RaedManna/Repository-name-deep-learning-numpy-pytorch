"""Small synthetic dataset utilities.

I used a two-moons style dataset because it is simple, visual, and not perfectly
linearly separable. That makes it a good sanity check for a small MLP.
"""

from __future__ import annotations

import numpy as np


def make_two_moons(
    n_samples: int = 500,
    noise: float = 0.12,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a simple two-moons binary classification dataset.

    This avoids depending on scikit-learn and keeps the project lightweight.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")

    rng = np.random.default_rng(seed)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    outer_angles = rng.uniform(0, np.pi, n_outer)
    inner_angles = rng.uniform(0, np.pi, n_inner)

    outer_x = np.c_[np.cos(outer_angles), np.sin(outer_angles)]
    inner_x = np.c_[1.0 - np.cos(inner_angles), 0.5 - np.sin(inner_angles)]

    X = np.vstack([outer_x, inner_x])
    y = np.concatenate([np.zeros(n_outer), np.ones(n_inner)]).reshape(-1, 1)

    X += rng.normal(scale=noise, size=X.shape)

    # Shuffle so the model does not see all examples from one class first.
    indices = rng.permutation(n_samples)
    return X[indices].astype(np.float32), y[indices].astype(np.float32)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train and test parts."""
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    test_size = int(len(X) * test_ratio)

    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
