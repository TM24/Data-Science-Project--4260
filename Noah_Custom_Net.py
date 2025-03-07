""" Noah Herron
    CSC 4260
    Custom Neural Net
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# getting the number of nodes in Hidden layer 1 and 2 from the user in command line
parser = argparse.ArgumentParser(description='Number of layers in the Network')
parser.add_argument('--layer1', type=int, default='20', help='Hidden Layer 1')
parser.add_argument('--layer2', type=int, default='20', help='Hidden Layer 2')
parser.add_argument('--iterations', type=int, default='550', help='Number of training iterations')
parser.add_argument('--alpha', type=float, default='0.1', help='Learning rate')
parser.add_argument('--x_train_path', type=str, default='C:/Path/To/Data/trainV4_combine.csv', help='Path to X_train CSV file')
parser.add_argument('--y_train_path', type=str, default='C:/Path/To/Data/DataV4_train_y.csv', help='Path to Y_train CSV file')
parser.add_argument('--x_test_path', type=str, default='X_test.csv', help='Path to X_test CSV file')
parser.add_argument('--y_test_path', type=str, default='Y_test.csv', help='Path to Y_test CSV file')
parser.add_argument('--results_dir', type=str, default='results/MathNetResults', help='Directory to save results')
parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing if no test set provided')

args = parser.parse_args()

# Create results directory if it doesn't exist
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

# Load and preprocess the data
def load_data():
    print("Loading data...")
    
    # Load training data
    X_train_df = pd.read_csv(args.x_train_path)
    Y_train_df = pd.read_csv(args.y_train_path)
    
    # Remove ID column
    X_train_df = X_train_df.drop('ID', axis=1)
    Y_train_df = Y_train_df.drop('ID', axis=1)
    
    # Convert to numpy arrays
    X_train = X_train_df.values
    Y_train = Y_train_df.values
    
    # Check if test data is provided
    if os.path.exists(args.x_test_path) and os.path.exists(args.y_test_path):
        print("Loading test data from files...")
        X_test_df = pd.read_csv(args.x_test_path)
        Y_test_df = pd.read_csv(args.y_test_path)
        
        # Remove ID column
        X_test_df = X_test_df.drop('ID', axis=1)
        Y_test_df = Y_test_df.drop('ID', axis=1)
        
        X_test = X_test_df.values
        Y_test = Y_test_df.values
    else:
        # Split training data for validation if no test set provided
        print("No test data provided. Splitting training data...")
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_train, Y_train, test_size=args.test_size, random_state=42)
    
    # Normalize features (important for neural networks)
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero
    
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    
    print(f"Data loaded successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")
    
    # Instead of transposing for this format, we'll adjust our functions
    return X_train, Y_train, X_test, Y_test

# defining the initial parameters - adjusted for your data dimensions
def init_params():
    n_features = X_train.shape[1]
    n_classes = Y_train.shape[1]
    
    # Add L2 regularization
    W1 = np.random.randn(args.layer1, n_features) * 0.01
    W2 = np.random.randn(args.layer2, args.layer1) * 0.01
    W3 = np.random.randn(n_classes, args.layer2) * 0.01
    
    return W1, b1, W2, b2, W3, b3

# ReLU function
def ReLU(Z):
    return np.maximum(Z, 0)

# ReLU derivative
def ReLU_deriv(Z):
    return Z > 0

# softmax function
def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # For numerical stability
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

# Modified forward propagation to handle data in samples as rows format
def forward_prop(W1, b1, W2, b2, W3, b3, X):
    # X shape is (n_samples, n_features)
    # We need to transpose X for matrix multiplication
    Z1 = W1.dot(X.T) + b1  # Shape: (layer1, n_samples)
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# Modified to handle multi-class Y in (n_samples, n_classes) format
def compute_loss(A3, Y, class_weights=None):
    """
    Compute loss with optional class weights
    
    Parameters:
    A3: Predicted probabilities (shape: n_classes, n_samples)
    Y: True labels (shape: n_samples, n_classes)
    class_weights: Optional array of weights for each class
    """
    Y = Y.T  # Transpose to match A3 shape
    m = Y.shape[1]
    
    # If no class weights provided, use balanced weights
    if class_weights is None:
        # Calculate inverse of class frequencies
        class_counts = np.sum(Y, axis=1)
        class_weights = 1 / (class_counts + 1e-8)
        class_weights /= np.sum(class_weights)
    
    # Ensure class_weights is a column vector
    class_weights = class_weights.reshape(-1, 1)
    
    # Weighted cross-entropy loss
    epsilon = 1e-8
    # Multiply each sample's loss by its class weight
    individual_losses = -Y * np.log(A3 + epsilon)
    weighted_losses = individual_losses * class_weights
    
    loss = np.sum(weighted_losses) / m
    
    return loss

# Modified backward propagation to handle data format
def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    m = X.shape[0]
    Y = Y.T  # Transpose to match A3 shape
    
    dZ3 = A3 - Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X)  # X is already transposed in this context
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

# Update parameters (remains the same)
def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3

# Modified to handle multi-class predictions
def get_predictions(A):
    # A shape is (n_classes, n_samples)
    return np.argmax(A, axis=0)

# Modified to handle Y in (n_samples, n_classes) format
def get_accuracy(predictions, Y):
    # Convert Y from one-hot to class indices
    Y_indices = np.argmax(Y, axis=1)
    return np.mean(predictions == Y_indices)

# Modified gradient descent to track history
def gradient_descent(X_train, Y_train, X_test, Y_test, alpha, iterations):
    # Calculate class weights
    class_counts = np.sum(Y_train, axis=0)
    class_weights = 1 / (class_counts + 1e-8)
    class_weights /= np.sum(class_weights)
    
    print("Class Weights:", class_weights)
    
    W1, b1, W2, b2, W3, b3 = init_params()
    
    # To track training history
    history = {
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': []
    }
    
    for i in range(iterations):
        # Forward and backward propagation for training
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X_train)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X_train, Y_train)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        
        # Calculate metrics every 10 iterations
        if i % 10 == 0:
            # Training metrics
            train_predictions = get_predictions(A3)
            train_accuracy = get_accuracy(train_predictions, Y_train)
            train_loss = compute_loss(A3, Y_train, class_weights)
            
            # Validation metrics
            _, _, _, _, _, A3_test = forward_prop(W1, b1, W2, b2, W3, b3, X_test)
            test_predictions = get_predictions(A3_test)
            test_accuracy = get_accuracy(test_predictions, Y_test)
            test_loss = compute_loss(A3_test, Y_test, class_weights)
            
            # Store in history
            history['accuracy'].append(train_accuracy)
            history['val_accuracy'].append(test_accuracy)
            history['loss'].append(train_loss)
            history['val_loss'].append(test_loss)
            
            print(f"Iteration: {i}, Training Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    return W1, b1, W2, b2, W3, b3, history

# Modified to return probability matrix
def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    # Return both predicted classes and probabilities
    predictions = get_predictions(A3)
    return predictions, A3.T  # Return probabilities in (n_samples, n_classes) format

# New function to plot training history
def plot_training_history(history):
    """
    Plot training metrics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
   
    # Convert iteration indices to epochs (assuming 10 iterations per "epoch" for plotting)
    epochs = range(len(history['accuracy']))
    
    ax1.plot(epochs, history['accuracy'], label='Training Accuracy')
    ax1.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch (x10 iterations)')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
   
    ax2.plot(epochs, history['loss'], label='Training Loss')
    ax2.plot(epochs, history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch (x10 iterations)')
    ax2.set_ylabel('Loss')
    ax2.legend()
   
    plt.tight_layout()
   
    # Save plot
    plt.savefig(f'{args.results_dir}/training_history.png')
    plt.close()
    print(f"Training history plot saved to {args.results_dir}/training_history.png")

# New function to evaluate model with confusion matrix
def evaluate_model(W1, b1, W2, b2, W3, b3, X_val, Y_val, class_names=['Home Win', 'Draw', 'Away Win']):
    """
    Evaluate model performance
    """
    # Get predictions
    # Capture all 6 return values from forward_prop
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X_val)
    y_pred_probs = A3.T  # Shape: (n_samples, n_classes)
    y_pred_classes = get_predictions(A3)
    
    # Convert one-hot Y_val to class indices
    y_val_classes = np.argmax(Y_val, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_val_classes, y_pred_classes, target_names=class_names))
   
    # Plot confusion matrix
    cm = confusion_matrix(y_val_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{args.results_dir}/confusion_matrix.png')
    plt.close()
    print(f"Confusion matrix saved to {args.results_dir}/confusion_matrix.png")
    
    return y_pred_probs

# Function to save predictions
def save_predictions(predictions, output_file=None):
    """
    Save predictions to CSV
    """
    if output_file is None:
        output_file = f'{args.results_dir}/predictions.csv'
        
    # Create DataFrame with class probabilities
    columns = ['Home_Win_Prob', 'Draw_Prob', 'Away_Win_Prob']
    
    if predictions.shape[1] == 4:  # If we have 4 columns
        columns = ['ID_Prob'] + columns
    
    pred_df = pd.DataFrame(predictions, columns=columns)
    
    # Add predicted class
    pred_df['Predicted_Outcome'] = np.argmax(predictions[:, -3:], axis=1)  # Ignore ID column if present
    
    # Map predicted outcome to match result
    outcome_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
    pred_df['Match_Result'] = pred_df['Predicted_Outcome'].map(outcome_map)
    
    # Save to CSV
    pred_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Main execution
if __name__ == "__main__":
    print("Starting neural network training...")
    
    # Load data
    X_train, Y_train, X_test, Y_test = load_data()
    
    # Train the model
    W1, b1, W2, b2, W3, b3, history = gradient_descent(X_train, Y_train, X_test, Y_test, args.alpha, args.iterations)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    class_names = ['Home Win', 'Draw', 'Away Win']
    test_predictions = evaluate_model(W1, b1, W2, b2, W3, b3, X_test, Y_test, class_names)
    
    # Save predictions
    save_predictions(test_predictions)
    
    print("\nTraining and evaluation complete!")