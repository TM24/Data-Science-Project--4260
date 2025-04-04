# Data Cleaning Notebook

## Overview
This notebook is designed to clean and preprocess large datasets for our project. Due to the volume of data, execution time may vary based on hardware capabilities.

## Recommendations
- Close all unnecessary applications before running this notebook unless your machine has high-performance specifications.
- Modify the parameters as needed to experiment with different data cleaning methods.

## Functionality
The notebook performs the following key operations:

1. **Load and Clean Data**
   - Reads CSV files and removes specific columns:
     - Standard deviation columns (`_std`)
     - Season average statistics (`_season_average`)
     - Last 5 match sum statistics (`_5_last_match_sum`)
   - Replaces NULL or NaN values with zero.

2. **Reshape Player Data**
   - Converts multiple rows per player ID into a single row.
   - Uses `pivot_table` to restructure player statistics.
   
3. **Further Cleaning**
   - Removes unnecessary player statistics:
     - Players 10-27
     - Team name, league, and player names
     - Position column

4. **Merge Team and Player Data**
   - Merges cleaned team data with reshaped player data.
   - Saves cleaned datasets to specified paths.
   - Frees memory using garbage collection (`gc`).

## Dataset Paths
Ensure that the raw data is located in the correct paths before execution:

- Training Data:
  - `train_home_team_statistics_df.csv`
  - `train_home_player_statistics_df.csv`
  - `train_away_team_statistics_df.csv`
  - `train_away_player_statistics_df.csv`
- Testing Data:
  - `test_home_team_statistics_df.csv`
  - `test_home_player_statistics_df.csv`
  - `test_away_team_statistics_df.csv`
  - `test_away_player_statistics_df.csv`

## Output
The cleaned and merged data is saved in:
- `train_home.csv`
- `train_away.csv`
- `test_home.csv`
- `test_away.csv`
- `trainV5.1_combine.csv`
- `testV5.1_combine.csv`

## Running the Notebook
1. Set the correct paths for your dataset.
2. Run the cells sequentially.
3. Verify the output files are generated correctly.

This notebook automates the data cleaning process to prepare high-quality datasets for further analysis and modeling.


