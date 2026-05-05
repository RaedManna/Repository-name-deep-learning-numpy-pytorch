# Deep Learning with NumPy and PyTorch

A clean portfolio project for practicing neural networks, binary classification, and deep learning fundamentals using both a from-scratch NumPy implementation and a PyTorch implementation.

The project focuses on understanding how a small neural network works internally: activation functions, forward propagation, binary cross-entropy loss, backpropagation, gradient descent, and model evaluation.

---

## Overview

This repository contains two versions of the same learning problem:

1. **NumPy MLP from scratch**  
   A small two-layer neural network implemented manually using NumPy.

2. **PyTorch MLP**  
   The same idea implemented using PyTorch to compare a manual implementation with a modern deep learning framework.

The goal is not to build the largest model, but to clearly understand the mechanics behind neural network training.

---

## What This Project Demonstrates

- Activation functions and derivatives
- Forward propagation
- Binary cross-entropy loss
- Backpropagation
- Gradient descent
- Mini-batch style training logic
- PyTorch model definition and training
- Accuracy evaluation
- Clean project organization
- Basic testing of activation functions and model behavior

---

## Project Structure

```text
deep-learning-numpy-pytorch/
│
├── src/
│   ├── activations.py
│   ├── data.py
│   ├── mlp_numpy.py
│   └── mlp_torch.py
│
├── examples/
│   └── run_experiment.py
│
├── tests/
│   └── test_activations.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech Stack

**Language:** Python  
**Core Libraries:** NumPy, PyTorch  
**Optional Visualization:** Matplotlib  
**Concepts:** Neural Networks, MLP, Backpropagation, Binary Classification, Deep Learning Fundamentals  

---

## How to Run

Install the requirements:

```bash
pip install -r requirements.txt
```

Run the example experiment:

```bash
python examples/run_experiment.py
```

Run the tests:

```bash
python -m pytest tests
```

---

## Notes

This repository is a cleaned portfolio-style project. It does not include raw exam submissions, assignment prompts, grading material, student identifiers, or private course files.

---

## Author

**Raed H. Manna**  
Computer Engineering Graduate | Junior Full-Stack Developer  

- GitHub: [RaedManna](https://github.com/RaedManna)
- LinkedIn: [raedhmanna](https://www.linkedin.com/in/raedhmanna)
