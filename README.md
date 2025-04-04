# Football Match Prediction Challenge

## Overview
This repository contains our group's submission for the **QRT Data Challenge**, which involves predicting the outcomes of football matches using machine learning models. Our work is based on extensive **Exploratory Data Analysis (EDA)**, **Poisson Regression**, **Predictive Factors Analysis**, and a **Neural Network Model**.

## Challenge Context
Professional sports, including football, have increasingly embraced data analytics for decision-making. This challenge focuses on **predicting match outcomes** using historical player and team statistics. The dataset is provided by Sportmonks and contains aggregated statistics from various football leagues worldwide.

## Challenge Goals
The objective is to develop a robust predictive model that works across different football leagues. The target variable is a **vector indicating match outcomes**:
- `[1,0,0]` → Home team wins
- `[0,1,0]` → Draw
- `[0,0,1]` → Away team wins

## Dataset Description
The dataset consists of:
- **Input Data** (`data/train_combine.csv`, `data/test_combine.csv`):
  - Home and Away team statistics, aggregated over the last 5 matches and season-to-date.
  - Player-level statistics.
- **Output Data** (`data/Y_train.csv`, `data/Y_test.csv`):
  - Match outcomes and additional targets such as goal difference.

## Project Structure
```
📂 repository
│── 📂 data
│   ├── Data_Files.csv
│── 📂 noah
│   ├── neural_networks.ipynb
│   ├── README.md (Detailed cleaning process)
│   ├── data_cleaning_notebook.ipynb
│── poisson_regression.ipynb
│── predictive_factors.ipynb
│── jair_eda.ipynb
│── README.md (This file)
```

## Methodologies Used
### 1. **Data Cleaning**
- Removed unnecessary columns (`_std`, `_season_average`, `5_last_match_sum`).
- Reshaped player data to match a single-row format per ID.
- Merged team and player data for final training datasets.

### 2. **Exploratory Data Analysis (EDA)**
- Examined statistical distributions of features.
- Identified key predictors influencing match outcomes.
- Visualized trends in team and player performance.

### 3. **Poisson Regression**
- Modeled football match scores based on past performance.
- Estimated probabilities of different match outcomes.

### 4. **Predictive Factors Model**
- Selected key variables using feature importance analysis.
- Applied statistical methods to refine feature selection.

### 5. **Neural Network Model**
- Implemented a multi-layer perceptron (MLP) neural network.
- Used training data to predict match results with probabilistic output.

## Results
- Each model's performance is evaluated using **accuracy metrics**.
- Final predictions are saved in `submissions/final_predictions.csv`.

## Running the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo.git
   ```
2. Run the models individually:
   ```bash
   jupyter notebook model.ipynb
   ```
   (Repeat for other models as needed)


## Contributors
- **[Noah Herron]** - Neural Network Model
- **[Jair Sanchez-Chavez]** -
- **[Emmanuel Hassan]** -
- **[Brady Reynolds]** -

---

## Disclaimer
The dataset is strictly for use within this challenge. Usage outside this scope is prohibited as per the challenge's terms of service.