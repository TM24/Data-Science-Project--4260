""" Noah Herron
    CSC 4260
    TensorFlow Neural Net
"""
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
def load_data():
    X_train = pd.read_csv('C:/Users/noahh/4260FinalProjectData/cleanedData/CleanV5/trainV5.3_combine.csv')
    X_test = pd.read_csv('C:/Users/noahh/4260FinalProjectData/cleanedData/CleanV5/testV5.3_combine.csv')
    Y_train = pd.read_csv('C:/Users/noahh/4260FinalProjectData/cleanedData/CleanV5/Y_train.csv')
    Y_test = pd.read_csv('C:/Users/noahh/4260FinalProjectData/cleanedData/CleanV5/Y_test.csv')
    return X_train, X_test, Y_train, Y_test

def improved_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Prepare data
def prepare_data(X_train, X_test, Y_train, Y_test):
    # Drop ID column from inputs
    X_train = X_train.drop(columns=['ID'], errors='ignore')
    X_test = X_test.drop(columns=['ID'], errors='ignore')

    # Drop ID column from labels
    Y_train = Y_train.drop(columns=['ID'], errors='ignore')
    Y_test = Y_test.drop(columns=['ID'], errors='ignore')

    # Scale input features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, Y_train, Y_test

# Evaluate and log to file
def evaluate_model(model, X_val, y_val, output_file):
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_val_classes = np.argmax(y_val.values, axis=1)

    report = classification_report(y_val_classes, y_pred_classes, target_names=['Home Win', 'Tie', 'Away Win'])
    print("\nClassification Report:")
    print(report)

    os.makedirs('results', exist_ok=True)
    with open(output_file, 'a') as f:
        f.write("\nClassification Report:\n")
        f.write(report + '\n')

    cm = confusion_matrix(y_val_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Home Win', 'Tie', 'Away Win'],
                yticklabels=['Home Win', 'Tie', 'Away Win'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('results/newNetConfusion_matrix.png')
    plt.close()

def plot_history(history):
    import matplotlib.pyplot as plt
    from datetime import datetime

    hist = history.history if hasattr(history, 'history') else history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/training_history_plots_{timestamp}.png"

    plt.figure(figsize=(12, 6))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(hist.get('loss', []), label='Training Loss')
    if 'val_loss' in hist:
        plt.plot(hist['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy and F1
    plt.subplot(1, 2, 2)
    plt.plot(hist.get('accuracy', []), label='Training Accuracy')
    if 'val_accuracy' in hist:
        plt.plot(hist['val_accuracy'], label='Validation Accuracy')
    if 'f1_score' in hist:
        plt.plot(hist['f1_score'], label='Training F1')
    if 'val_f1_score' in hist:
        plt.plot(hist['val_f1_score'], label='Validation F1')
    plt.title('Accuracy / F1 over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Training history plot saved to: {filename}")

def main():
    print("Loading data...")
    X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = load_data()

    print("Label distribution:")
    print(Y_train_raw.sum())

    print("Preparing data...")
    X_train_scaled, X_test_scaled, Y_train, Y_test = prepare_data(X_train_raw, X_test_raw, Y_train_raw, Y_test_raw)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_scaled, Y_train, test_size=0.2, random_state=42
    )

    y_train_labels = np.argmax(y_train.values, axis=1)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_labels), y=y_train_labels)
    class_weight_dict = dict(enumerate(class_weights))
    print("Class Weights:", class_weight_dict)

    print("Training improved model...")
    model = improved_model(input_shape=(X_train_scaled.shape[1],))

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint("results/best_model.keras", save_best_only=True, monitor='val_loss')

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    print("Evaluating on validation set...")
    evaluate_model(model, X_val, y_val, 'results/newNetValidation_report.txt')

    print("Evaluating on test set...")
    evaluate_model(model, X_test_scaled, Y_test, 'results/newNetTest_report.txt')

    print("Saving predictions...")
    predictions = model.predict(X_test_scaled)
    pred_df = pd.DataFrame(predictions, columns=['Home_Win_Prob', 'Draw_Prob', 'Away_Win_Prob'])
    pred_df['Predicted_Outcome'] = np.argmax(predictions, axis=1)
    pred_df['True_Outcome'] = np.argmax(Y_test.values, axis=1)
    pred_df.to_csv('results/newNetPredictions.csv', index=False)

    wrong_preds = pred_df[pred_df['Predicted_Outcome'] != pred_df['True_Outcome']]
    wrong_preds.to_csv('results/misclassified.csv', index=False)

    print("Saving model...")
    model.save('results/newNetSoccer_outcome_model.keras')

    print("Done.")
    plot_history(history)

if __name__ == "__main__":
    main()