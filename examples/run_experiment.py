"""Run a small NumPy vs PyTorch experiment.

This script is meant to be simple to read. It creates a two-moons style dataset,
trains a manual NumPy neural network, then trains a similar PyTorch model.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allows running the example without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import make_two_moons, train_test_split
from src.mlp_numpy import NumpyMLP
from src.mlp_torch import train_torch_model


def main() -> None:
    X, y = make_two_moons(n_samples=700, noise=0.12, seed=7)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.25, seed=7)

    print("Training NumPy model from scratch...")
    numpy_model = NumpyMLP(input_dim=2, hidden_dim=20, seed=7)
    numpy_model.fit(X_train, y_train, epochs=1500, lr=0.08, print_every=300)

    train_acc = numpy_model.accuracy(X_train, y_train)
    test_acc = numpy_model.accuracy(X_test, y_test)
    print(f"NumPy final train accuracy: {train_acc:.3f}")
    print(f"NumPy final test accuracy : {test_acc:.3f}")

    print("\nTraining PyTorch model...")
    train_torch_model(
        X_train,
        y_train,
        X_test,
        y_test,
        hidden_dim=20,
        epochs=1000,
        lr=0.03,
        print_every=250,
        seed=7,
    )


if __name__ == "__main__":
    main()
