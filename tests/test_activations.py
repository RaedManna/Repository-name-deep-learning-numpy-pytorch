import numpy as np

from src.activations import (
    elu,
    leaky_relu,
    relu,
    sigmoid,
    sigmoid_derivative,
    tanh,
)


def test_relu_sets_negative_values_to_zero():
    x = np.array([-2.0, -0.5, 0.0, 3.0])
    out = relu(x)
    assert np.allclose(out, np.array([0.0, 0.0, 0.0, 3.0]))


def test_sigmoid_output_range():
    x = np.array([-100.0, 0.0, 100.0])
    out = sigmoid(x)
    assert np.all(out >= 0)
    assert np.all(out <= 1)
    assert np.isclose(out[1], 0.5)


def test_sigmoid_derivative_at_zero():
    x = np.array([0.0])
    assert np.allclose(sigmoid_derivative(x), np.array([0.25]))


def test_tanh_matches_numpy():
    x = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(tanh(x), np.tanh(x))


def test_leaky_relu_keeps_small_negative_slope():
    x = np.array([-2.0, 2.0])
    out = leaky_relu(x, alpha=0.1)
    assert np.allclose(out, np.array([-0.2, 2.0]))


def test_elu_negative_value():
    x = np.array([-1.0])
    out = elu(x, alpha=1.0)
    assert out[0] < 0
