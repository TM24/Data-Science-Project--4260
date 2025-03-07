""" Noah Herron
    CSC 4260
    TensorFlow Neural Net
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import os
import logging
from datetime import datetime

def load_data():
    """
    Load all required datasets with your new filenames
    """
    try:
        X_train_home = pd.read_csv('C:/Path/To/Data/DataV4_train_home.csv')
        X_train_away = pd.read_csv('C:/Path/To/Data/DataV4_train_away.csv')
        X_test_home = pd.read_csv('C:/Path/To/Data/DataV4_test_home.csv')
        X_test_away = pd.read_csv('C:/Path/To/Data/DataV4_test_away.csv')
        Y_train = pd.read_csv('C:/Path/To/Data/DataV4_train_y.csv')
        Y_test = pd.read_csv('C:/Path/To/Data/DataV4_test_y.csv')
        
        return X_train_home, X_train_away, X_test_home, X_test_away, Y_train, Y_test
    
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        exit(1)

def setup_logging():
    """
    Set up logging configuration for summary statistics only
    """
    os.makedirs('results/smallResults/logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'results/smallResults/logs/model_summary_{timestamp}.log'
    
    # Configure logging to write only to file (not console)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',  # Simplified format without timestamp
        handlers=[
            logging.FileHandler(log_filename)
        ]
    )
    return log_filename

def log_final_evaluation(model, X_test, y_test):
    """
    Log only the final model evaluation metrics
    """
    # Get test set metrics
    test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=0)
    
    # Get predictions and classification report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    report = classification_report(y_test_classes, y_pred_classes, 
                                 target_names=['Home Win', 'Draw', 'Away Win'])
    
    # Log the summary statistics
    logging.info("Classification Report:")
    logging.info(report)
    logging.info("\nTest Set Metrics:")
    logging.info(f"Loss: {test_loss:.4f}")
    logging.info(f"Accuracy: {test_accuracy:.4f}")
    logging.info(f"AUC: {test_auc:.4f}")

# def engineer_features(team_stats, player_stats):
#     """
#     Engineer features by combining team and player statistics
#     """
#     # Aggregate player stats by team (mean and max, no std since it's removed)
#     if not player_stats.empty:
#         player_aggs = player_stats.groupby('ID').agg({
#             col: ['mean', 'max'] for col in player_stats.select_dtypes(include=[np.number]).columns
#             if col != 'ID'
#         }).fillna(0)
        
#         # Flatten column names
#         player_aggs.columns = [f'PLAYER_{a}_{b}' for a, b in player_aggs.columns]
#         player_aggs = player_aggs.reset_index()
#     else:
#         player_aggs = pd.DataFrame({'ID': team_stats['ID']})
    
#     # Copy team stats for feature engineering
#     team_features = team_stats.copy()

#     # # New feature: Attack efficiency based on 5 last match average
#     # if 'TEAM_GOALS_5_last_match_average' in team_stats.columns and 'TEAM_DANGEROUS_ATTACKS_5_last_match_average' in team_stats.columns:
#     #     team_features['ATTACK_EFFICIENCY_5_MATCH'] = (
#     #         team_stats['TEAM_GOALS_5_last_match_average'] / 
#     #         team_stats['TEAM_DANGEROUS_ATTACKS_5_last_match_average'].clip(lower=1)
#     #     )
    
#     # New feature: Average goals per match
#     if 'TEAM_GOALS_5_last_match_average' in team_stats.columns:
#         team_features['AVG_GOALS'] = team_stats['TEAM_GOALS_5_last_match_average']

#     # Merge team and player stats
#     combined_features = pd.merge(team_features, player_aggs, on='ID', how='left')

#     return combined_features


def create_enhanced_model(input_shape):
    """
    Create a neural network with 128-neuron layers
    """
    inputs = layers.Input(shape=input_shape)
    
    # First layer
    x = layers.Dense(128, activation='relu')(inputs)  # Changed to 128 neurons
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Two residual blocks with 128 neurons each
    for _ in range(2):
        y = layers.Dense(128, activation='relu')(x)  # Changed to 128 neurons
        y = layers.BatchNormalization()(y)
        y = layers.Dropout(0.2)(y)
        y = layers.Dense(128, activation='relu')(y)  # Changed to 128 neurons
        y = layers.BatchNormalization()(y)
        
        # Direct connection (no dimension matching needed since same size)
        x = layers.Add()([x, y])
        x = layers.Activation('relu')(x)
    
    # Final 40-neuron layer
    x = layers.Dense(128, activation='relu')(x)  # Changed to 128 neurons
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    # Output layer (still 3 outputs for three possible outcomes)
    outputs = layers.Dense(3, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Use ExponentialDecay for learning rate
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

def prepare_data(X_train_home, X_train_away, X_test_home, X_test_away, Y_train, Y_test):
    """
    Prepare and scale the data by combining home and away stats
    """
    X_train_combined = pd.concat([X_train_home, X_train_away], axis=1)
    X_test_combined = pd.concat([X_test_home, X_test_away], axis=1)
    
    # Remove non-numerical columns
    columns_to_drop = ['ID', 'LEAGUE', 'TEAM_NAME', 'PLAYER_NAME', 'POSITION']
    train_features = X_train_combined.drop([col for col in columns_to_drop if col in X_train_combined.columns], axis=1)
    test_features = X_test_combined.drop([col for col in columns_to_drop if col in X_test_combined.columns], axis=1)
    Y_train_features = Y_train.drop([col for col in columns_to_drop if col in Y_train.columns], axis=1)
    Y_test_features = Y_test.drop([col for col in columns_to_drop if col in Y_test.columns], axis=1)
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_features)
    X_test_scaled = scaler.transform(test_features)
    
    return X_train_scaled, X_test_scaled, Y_train_features, Y_test_features

def plot_training_history(history):
    """
    Plot training metrics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('results/smallResults/training_history.png')
    plt.close()

def evaluate_model(model, X_val, y_val, class_names=['Home Win', 'Draw', 'Away Win']):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)
    
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
    plt.savefig('results/smallResults/confusion_matrix.png')
    plt.close()

def save_predictions(predictions, output_file='predictions.csv'):
    """
    Save predictions to CSV
    """
    pred_df = pd.DataFrame(predictions, columns=['Home_Win_Prob', 'Draw_Prob', 'Away_Win_Prob'])
    pred_df['Predicted_Outcome'] = np.argmax(predictions, axis=1)
    pred_df.to_csv(output_file, index=False)

    
def main():
    # Set up logging at the end when we have results
    log_filename = setup_logging()
    
    # Create output directory for results
    os.makedirs('results', exist_ok=True)
    
    # Load data with Y_test
    print("Loading data...")
    X_train_home, X_train_away, X_test_home, X_test_away, Y_train, Y_test = load_data()
    
    # Debug to show the data
    print("Sample of home team stats:")
    print(X_train_home.head())
    
    # Add these debug lines here
    print("Y_train columns:", Y_train.columns)
    print("Y_test columns:", Y_test.columns)
    print("First few rows of Y_train:")
    print(Y_train.head())
    print("First few rows of Y_test:")
    print(Y_test.head())
    
    # Prepare data
    print("Preparing data...")
    X_train_scaled, X_test_scaled, Y_train_Prep, Y_test_Prep = prepare_data(X_train_home, X_train_away, X_test_home, X_test_away, Y_train, Y_test)
    
    # Debug: Print shapes of scaled data and labels
    print("X_train_scaled shape:", X_train_scaled.shape)
    print("X_test_scaled shape:", X_test_scaled.shape)
    print("Y_train shape:", Y_train_Prep.shape)
    print("Y_test shape:", Y_test_Prep.shape)
    
    # NOT NECCESARY START
    # # Convert Y_train to categorical
    # y_train_numerical = np.zeros(len(Y_train))
    # y_train_numerical[Y_train['DRAW'] == 1] = 1
    # y_train_numerical[Y_train['AWAY_WINS'] == 1] = 2
    # y_train_encoded = tf.keras.utils.to_categorical(y_train_numerical)
    
    # # Convert Y_test to categorical
    # y_test_numerical = np.zeros(len(Y_test))
    # y_test_numerical[Y_test['DRAW'] == 1] = 1
    # y_test_numerical[Y_test['AWAY_WINS'] == 1] = 2
    # y_test_encoded = tf.keras.utils.to_categorical(y_test_numerical)
    # NOT NECCESARY END
    
    # # Debug: Print shapes of encoded data
    # print("y_train_encoded shape:", y_train_encoded.shape)
    # print("y_test_encoded shape:", y_test_encoded.shape)
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_scaled, Y_train_Prep, test_size=0.2, random_state=42
    )
    
    # Debug: Print shapes after splitting
    print("After split:")
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:", y_val.shape)
    
    # print("Y_train column names", y_train.columns)

    
    # debug to ensure right shape
    # print(X_train.shape[1])
    
    # Create and train model
    print("Training model...")
    model = create_enhanced_model(input_shape=(X_train.shape[1],))
    
    # callbacks_list = [
    #     callbacks.EarlyStopping(
    #         monitor='val_loss',
    #         patience=20,
    #         restore_best_weights=True
    #     )
    # ]
    
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_val, y_val),
        # callbacks=callbacks_list,
        verbose=1
    )
    
    # Evaluate model on validation set
    print("Evaluating model on validation set...")
    plot_training_history(history)
    evaluate_model(model, X_val, y_val)
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy, test_auc = model.evaluate(X_test_scaled, Y_test_Prep, verbose=1)
    print(f"\nTest Set Metrics:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"AUC: {test_auc:.4f}")
    
    # Generate detailed test set evaluation
    print("\nGenerating detailed test set evaluation...")
    evaluate_model(model, X_test_scaled, Y_test_Prep)
    
    # After model training and evaluation is complete, log the final results
    log_final_evaluation(model, X_test_scaled, Y_test_Prep)
    print(f"\nSummary statistics have been saved to: {log_filename}")
    
    # Make predictions on test set
    print("Making predictions...")
    predictions = model.predict(X_test_scaled)
    
    # Save predictions with true values
    pred_df = pd.DataFrame(predictions, columns=['Home_Win_Prob', 'Draw_Prob', 'Away_Win_Prob'])
    pred_df['Predicted_Outcome'] = np.argmax(predictions, axis=1)
    pred_df['True_Outcome'] = np.argmax(Y_test, axis=1)
    pred_df.to_csv('results/predictions.csv', index=False)
    
    # Save model
    model.save('results/match_prediction_model.keras')
    
    print("\nTraining completed successfully!")
    print("Results have been saved to the 'results' directory:")
    print("- Model: results/match_prediction_model.keras")
    print("- Predictions: results/predictions.csv")
    print("- Training history plot: results/training_history.png")
    print("- Confusion matrix: results/confusion_matrix.png")

if __name__ == "__main__":
    main()
