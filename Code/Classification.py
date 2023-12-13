# Import packages
import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
tf.random.set_seed(42)
from joblib import dump


# Function to write model information to file
def write_model_info_to_file(file_path, model_name, execution_time, best_params, classification_report):
    with open(file_path, 'a') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Execution time: {execution_time:.2f} seconds\n")
        f.write(f"Best parameters: {best_params}\n")
        f.write("Classification report:\n")
        f.write(f"{classification_report}\n")
        f.write("-" * 80 + "\n\n")

# Function to read data and split into train and test sets
def load_and_split_data(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "LN_labeled_data.csv"), index_col=0, header=0)
    X = df.drop('Population', axis=1)
    y = df['Population']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and evaluate sklearn ML models
def train_and_evaluate_model(model_pipeline, param_grid, X_train, y_train, X_test, y_test, model_name, file_path, models_directory):
    start_time = time.time()
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.reset_index(inplace=True)
    df_report.rename(columns={'index': 'Cell Type'}, inplace=True)

    execution_time = time.time() - start_time
    write_model_info_to_file(file_path, model_name, execution_time, grid_search.best_params_, df_report.to_string(index=False))

    # Save the best model
    model_file_name = f'best_model_{model_name}.joblib'
    dump(grid_search, os.path.join(models_directory, model_file_name))

    return grid_search

# Function to create the Neural Network model
def train_nn_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size):
    NN_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    NN_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr), metrics=['accuracy'])
    NN_model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.2, verbose=0)
    return NN_model

# Function to train and evaluate the neural network
def train_and_evaluate_nn(X_train, y_train, X_test, y_test, nn_param_grid, file_path, models_directory):
    start_time = time.time()

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_train_one_hot = to_categorical(y_train_encoded, num_classes=5)
    y_test_one_hot = to_categorical(y_test_encoded, num_classes=5)

    scaler_NN = MinMaxScaler()
    X_train_scaled = scaler_NN.fit_transform(X_train)
    X_test_scaled = scaler_NN.transform(X_test)

    least_val_loss = float('inf')
    least_loss_model = None
    best_params = None

    for num_nodes in nn_param_grid['num_nodes']:
        for dropout_prob in nn_param_grid['dropout_prob']:
            for lr in nn_param_grid['lr']:
                for batch_size in nn_param_grid['batch_size']:
                    model = train_nn_model(X_train_scaled, y_train_one_hot, num_nodes, dropout_prob, lr, batch_size)
                    val_loss, _ = model.evaluate(X_test_scaled, y_test_one_hot, verbose=0)
                    if val_loss < least_val_loss:
                        least_val_loss = val_loss
                        least_loss_model = model
                        best_params = (num_nodes, dropout_prob, lr, batch_size)

    y_pred_probs = least_loss_model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_probs, axis=1)
    report = classification_report(y_test_encoded, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.reset_index(inplace=True)
    df_report.rename(columns={'index': 'Cell Type'}, inplace=True)

    execution_time = time.time() - start_time
    write_model_info_to_file(file_path, "Neural Network", execution_time, best_params, df_report.to_string(index=False))

    # Save the best model
    least_loss_model.save(os.path.join(models_directory, 'best_model_NN'))

# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <path_to_main_directory>")
        sys.exit(1)

    main_directory = sys.argv[1]
    data_directory = os.path.join(main_directory, "Data")
    models_directory = os.path.join(main_directory, "Models")
    file_path = os.path.join(models_directory, 'best_models_parameters.txt')

    # Open the file in write mode to create an empty file or clear it if it already exists
    with open(file_path, 'w') as file:
        pass

    # Load and preprocess data
    df = pd.read_csv(os.path.join(data_directory, "LN_labeled_data.csv"), index_col=0, header=0)
    X = df.drop('Population', axis=1)
    y = df['Population']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate models

    # Logistic Regression
    pipeline_lr = Pipeline([
        ('scaler', MinMaxScaler()),
        ('logreg', LogisticRegression(max_iter=1000))
    ])
    param_grid_lr = {
        'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'logreg__penalty': ['l1', 'l2'],
        'logreg__solver': ['liblinear']
    }
    train_and_evaluate_model(pipeline_lr, param_grid_lr, X_train, y_train, X_test, y_test, "Logistic_Regression", file_path, models_directory)

    # K-Nearest Neighbors
    pipeline_kNN = Pipeline([
        ('scaler', MinMaxScaler()),
        ('knn', KNeighborsClassifier())
    ])
    param_grid_kNN = {
        'knn__n_neighbors': [3, 5, 7, 9],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan']
    }
    train_and_evaluate_model(pipeline_kNN, param_grid_kNN, X_train, y_train, X_test, y_test, "K-Nearest_Neighbors", file_path, models_directory)

    # Naive Bayes
    pipeline_NB = Pipeline([
        ('scaler', MinMaxScaler()),
        ('nb', GaussianNB())
    ])
    param_grid_NB = {
        'nb__var_smoothing': [1e-09, 1e-08, 1e-07, 1e-06]
    }
    train_and_evaluate_model(pipeline_NB, param_grid_NB, X_train, y_train, X_test, y_test, "Naive_Bayes", file_path, models_directory)

    # Random Forest
    pipeline_RF = Pipeline([
        ('scaler', MinMaxScaler()),
        ('rf', RandomForestClassifier())
    ])
    param_grid_RF = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_features': ['sqrt', 'log2'],
        'rf__max_depth': [None, 5, 10, 20]
    }
    train_and_evaluate_model(pipeline_RF, param_grid_RF, X_train, y_train, X_test, y_test, "Random_Forest", file_path, models_directory)

    # Neural Network
    nn_param_grid = {
        'num_nodes': [8, 16, 32],
        'dropout_prob': [0, 0.2],
        'lr': [0.01, 0.005, 0.001],
        'batch_size': [32, 64, 128]
    }
    train_and_evaluate_nn(X_train, y_train, X_test, y_test, nn_param_grid, file_path, models_directory)

if __name__ == "__main__":
    main()
