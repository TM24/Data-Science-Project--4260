# Soccer Match Outcome Prediction Project

## Overview

This project aims to predict the outcome of soccer matches (Home Win, Tie, Away Win) using machine learning and deep learning models. It consists of:

- A **data cleaning notebook** to preprocess raw team and player statistics.
- A **custom neural network** implemented with NumPy (`Noah_Custom_Net.py`).
- A **deep learning model** using TensorFlow/Keras (`Noah_Tensorflow.py`).

---

## 1. Data Cleaning (`CleanData.ipynb`)

### Functionality

1. **Load and Clean Data**
   - Reads team and player statistics from CSV files.
   - Drops columns with standard deviation (`_std`), season averages (`_season_average`), and last 5 match sums (`_5_last_match_sum`).
   - Fills missing values with zeros.

2. **Reshape Player Data**
   - Uses `pivot_table` to reshape rows into single rows per match/player.

3. **Remove Unneeded Columns**
   - Drops data from players 10–27 and non-numerical identifiers like names and positions.

4. **Merge Cleaned Data**
   - Combines cleaned team and player statistics into unified datasets.
   - Saves processed datasets for model consumption (e.g., `trainV5.3_combine.csv`, `testV5.3_combine.csv`).

---

## 2. Custom NumPy Neural Network (`Noah_Custom_Net.py`)

### Description

A fully custom 3-layer neural network built from scratch using NumPy to classify soccer match outcomes.

### Features

- Command-line configurable hidden layer sizes (`--layer1`, `--layer2`).
- Uses ReLU and softmax activations.
- Includes manual forward/backward propagation and gradient descent.
- Trains on preprocessed CSV data.
- Prints training accuracy every 10 iterations.
- Evaluates performance on a dev set.

### Example Usage

```bash
python Noah_Custom_Net.py --layer1 64 --layer2 32
```

---

## 3. TensorFlow/Keras Model (`Noah_Tensorflow.py`)

### Description

A deep learning pipeline using Keras for more scalable training and evaluation.

### Features

- Automatically loads and preprocesses training/testing data.
- Applies MinMax scaling and one-hot encoding.
- Trains a deep feedforward neural network with:
  - 512, 256, and 128 neurons
  - Batch normalization and dropout
  - Early stopping and model checkpointing
- Generates:
  - Classification reports
  - Confusion matrix
  - Training history plots
  - Predictions and misclassified samples

### Output

All output is saved in the `results/` directory:

- `best_model.keras`: Best-performing model
- `training_history_plots_TIMESTAMP.png`: Loss/accuracy plots
- `newNetValidation_report.txt`, `newNetTest_report.txt`: Evaluation reports
- `newNetPredictions.csv`, `misclassified.csv`: Prediction results

---

## Dataset Requirements

Ensure the following files are available at the paths hardcoded or configured within the scripts:

- Cleaned Input:
  - `trainV5.3_combine.csv`, `testV5.3_combine.csv`
- Labels:
  - `Y_train.csv`, `Y_test.csv`

---

## How to Run

1. **Clean the raw data** using `CleanData.ipynb`.
2. **Train the custom model** using `python Noah_Custom_Net.py`.
3. **Run the TensorFlow model** using:

```bash
python Noah_Tensorflow.py
```

Make sure to adjust hardcoded paths or update the scripts to use relative paths if necessary.
