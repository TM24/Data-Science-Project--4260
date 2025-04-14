""" Noah Herron
    CSC 4260
    Custom Neural Net
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler

# getting the number of nodes in Hidden layer 1 and 2 from the user in command line
parser = argparse.ArgumentParser(description='Number of layers in the Network')
parser.add_argument('--layer1', type=int, default=20, help='Hidden Layer 1')
parser.add_argument('--layer2', type=int, default=20, help='Hidden Layer 2')

args = parser.parse_args()

# Load the football data
data_train = pd.read_csv('C:/Path/To/Data/trainV5.3_combine.csv')
data_test = pd.read_csv('C:/Path/To/Data/testV5.3_combine.csv')

# Load the labels separately
Y_train_data = pd.read_csv('C:/Path/To/Data/Y_train.csv')

# Extract class labels from one-hot columns (ignore ID column)
Y_train_raw_val = Y_train_data.iloc[:, 1:].values
Y_train_raw = np.argmax(Y_train_raw_val, axis=1)

# Remove ID column from input features
X_train_raw = data_train.iloc[:, 1:].values
X_test_raw = data_test.iloc[:, 1:].values

# Split into dev and training sets
X_dev = X_train_raw[:1000]
Y_dev = Y_train_raw[:1000]
X_train = X_train_raw[1000:]
Y_train = Y_train_raw[1000:]

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test_raw)

# Transpose for neural network
X_train = X_train.T
X_dev = X_dev.T

# Initialization
def init_params():
    W1 = np.random.rand(args.layer1, 157) - 0.5
    b1 = np.random.rand(args.layer1, 1) - 0.5
    W2 = np.random.rand(args.layer2, args.layer1) - 0.5
    b2 = np.random.rand(args.layer2, 1) - 0.5
    W3 = np.random.rand(3, args.layer2) - 0.5
    b3 = np.random.rand(3, 1) - 0.5
    return W1, b1, W2, b2, W3, b3

def ReLU(Z): return np.maximum(Z, 0)
def ReLU_deriv(Z): return Z > 0

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = ReLU(Z2)
    Z3 = W3 @ A2 + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 3))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    if A3.shape != one_hot_Y.shape:
        raise ValueError(f"Shape mismatch: A3 {A3.shape} vs Y {one_hot_Y.shape}")
    dZ3 = A3 - one_hot_Y
    dW3 = 1/m * dZ3 @ A2.T
    db3 = 1/m * np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = W3.T @ dZ3 * ReLU_deriv(Z2)
    dW2 = 1/m * dZ2 @ A1.T
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T @ dZ2 * ReLU_deriv(Z1)
    dW1 = 1/m * dZ1 @ X.T
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W3 -= alpha * dW3
    b3 -= alpha * db3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A): return np.argmax(A, axis=0)
def get_accuracy(preds, Y): return np.mean(preds == Y)

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 10 == 0:
            acc = get_accuracy(get_predictions(A3), Y)
            print(f"Iteration {i}: Accuracy {acc:.4f}")
    return W1, b1, W2, b2, W3, b3

def evaluate_model(W1, b1, W2, b2, W3, b3, X, Y):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    preds = get_predictions(A3)
    acc = get_accuracy(preds, Y)
    print(f"Model Accuracy: {acc*100:.2f}%")
    return acc

print("Training the model...")
W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, alpha=0.1, iterations=550)
print("\nEvaluating on dev set:")
evaluate_model(W1, b1, W2, b2, W3, b3, X_dev, Y_dev)
