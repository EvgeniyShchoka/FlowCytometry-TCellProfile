# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
tf.random.set_seed(42)

from joblib import dump

import time

import os
import sys

# Use command-line arguments to specify the main directory
if len(sys.argv) != 2:
    print("Usage: python script_name.py <path_to_main_directory>")
    sys.exit(1)

main_directory = sys.argv[1]

# Define paths for data and models
data_directory = os.path.join(main_directory, "Data")
models_directory = os.path.join(main_directory, "Models")

# define the path to the output summary file
file_path = os.path.join(models_directory, 'best_models_parameters.txt')

# open the file in write mode to create an empty file or clear it if it already exists
with open(file_path, 'w') as file:
    pass

# define a function to write the model information to the output file
def write_model_info_to_file(file_path, model_name, execution_time, best_params, classification_report):
    with open(file_path, 'a') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Execution time: {int(execution_time // 60)} minutes and {int(execution_time % 60)} seconds\n" if execution_time >= 60 else f"Execution time: {int(execution_time)} seconds\n")
        f.write(f"Best parameters: {best_params}\n")
        f.write("Classification report:\n")
        f.write(f"{classification_report}\n")
        f.write("-" * 80 + "\n\n")

# read csv table with rows and columns
df = pd.read_csv(os.path.join(data_directory, "LN_labeled_data.csv"), index_col=0, header=0)

# split the data into X and y
X = df.drop('Population', axis=1)
y = df['Population']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic regression

# start the timer
start_time_LR = time.time()

# create a pipeline with data standardization and logistic regression
pipeline_LR = Pipeline([
    ('scaler', MinMaxScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])

# define the hyperparameters to be optimized
param_grid_LR = {
    'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'logreg__penalty': ['l1', 'l2'],
    'logreg__solver': ['liblinear']
}

# create a GridSearchCV object
grid_search_LR = GridSearchCV(pipeline_LR, param_grid_LR, cv=5, scoring='accuracy', n_jobs=-1)

# fit the model
grid_search_LR.fit(X_train, y_train)

# keep best parameters
best_params = grid_search_LR.best_params_

# make prediction on the test set
y_pred = grid_search_LR.predict(X_test)

# generate the classification report
report = classification_report(y_test, y_pred, output_dict=True)

# convert the report to a DataFrame
df_report = pd.DataFrame(report).transpose()

# add the column with the cell types
df_report.reset_index(inplace=True)
df_report.rename(columns={'index': 'Cell Type'}, inplace=True)

# stop the timer
end_time_LR = time.time()

# calculate the execution time
duration_LR = end_time_LR - start_time_LR

write_model_info_to_file(file_path, "Logistic regression", duration_LR, best_params, df_report.to_string(index=False))


# K-Nearest Neighbor

# start the timer
start_time_kNN = time.time()

# create a pipeline with data standardization and KNN
pipeline_kNN = Pipeline([
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsClassifier())
])

# define the hyperparameters to be optimized
param_grid_kNN = {
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}

# create a GridSearchCV object
grid_search_kNN = GridSearchCV(pipeline_kNN, param_grid_kNN, cv=5, scoring='accuracy', n_jobs=-1)

# fit the model
grid_search_kNN.fit(X_train, y_train)

# keep best parameters
best_params = grid_search_kNN.best_params_

# make prediction on the test set
y_pred = grid_search_kNN.predict(X_test)

# generate the classification report
report = classification_report(y_test, y_pred, output_dict=True)

# convert the report to a DataFrame
df_report = pd.DataFrame(report).transpose()

# add the column with the cell types
df_report.reset_index(inplace=True)
df_report.rename(columns={'index': 'Cell Type'}, inplace=True)

# stop the timer
end_time_kNN = time.time()

# calculate the execution time
duration_kNN = end_time_kNN - start_time_kNN

write_model_info_to_file(file_path, "K-Nearest Neighbor", duration_kNN, best_params, df_report.to_string(index=False))


# Naive Bayes

# start the timer
start_time_NB = time.time()

# create a pipeline with data standardization and Naive Bayes
pipeline_NB = Pipeline([
    ('scaler', MinMaxScaler()),
    ('nb', GaussianNB())
])

# define the hyperparameters to be optimized
param_grid_NB = {
    'nb__var_smoothing': [1e-09, 1e-08, 1e-07, 1e-06]
}

# create a GridSearchCV object
grid_search_NB = GridSearchCV(pipeline_NB, param_grid_NB, cv=5, scoring='accuracy', n_jobs=-1)

# fit the model
grid_search_NB.fit(X_train, y_train)

# keep best parameters
best_params = grid_search_NB.best_params_

# make prediction on the test set
y_pred = grid_search_NB.predict(X_test)

# generate the classification report
report = classification_report(y_test, y_pred, output_dict=True)

# convert the report to a DataFrame
df_report = pd.DataFrame(report).transpose()

# add the column with the cell types
df_report.reset_index(inplace=True)

# stop the timer
end_time_NB = time.time()

# calculate the execution time
duration_NB = end_time_NB - start_time_NB

write_model_info_to_file(file_path, "Naive Bayes", duration_NB, best_params, df_report.to_string(index=False))


# Random Forest

# start the timer
start_time_RF = time.time()

# create a pipeline with data standardization and Random Forest
pipeline_RF = Pipeline([
    ('scaler', MinMaxScaler()),
    ('rf', RandomForestClassifier())
])

# define the hyperparameters to be optimized
param_grid_RF = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_features': ['sqrt', 'log2'],
    'rf__max_depth': [None, 5, 10, 20]
}

# create a GridSearchCV object
grid_search_RF = GridSearchCV(pipeline_RF, param_grid_RF, cv=5, scoring='accuracy', n_jobs=-1)

# fit the model
grid_search_RF.fit(X_train, y_train)

# keep best parameters
best_params = grid_search_RF.best_params_

# make prediction on the test set
y_pred = grid_search_RF.predict(X_test)

# generate the classification report
report = classification_report(y_test, y_pred, output_dict=True)

# convert the report to a DataFrame
df_report = pd.DataFrame(report).transpose()

# add the column with the cell types
df_report.reset_index(inplace=True)
df_report.rename(columns={'index': 'Cell Type'}, inplace=True)

# stop the timer
end_time_RF = time.time()

# calculate the execution time
duration_RF = end_time_RF - start_time_RF

write_model_info_to_file(file_path, "Random Forest", duration_RF, best_params, df_report.to_string(index=False))


# Neural Network

# start the timer
start_time_NN = time.time()

# splitting the data
X_train_NN, X_val_NN, y_train_NN, y_val_NN = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# label Encoding
label_encoder = LabelEncoder()
y_train_NN_encoded = label_encoder.fit_transform(y_train_NN)
y_val_NN_encoded = label_encoder.transform(y_val_NN)
y_test_NN_encoded = label_encoder.transform(y_test)

# one-Hot Encoding
y_train_NN_one_hot = to_categorical(y_train_NN_encoded, num_classes=5)
y_val_NN_one_hot = to_categorical(y_val_NN_encoded, num_classes=5)
y_test_NN_one_hot = to_categorical(y_test_NN_encoded, num_classes=5)

# training Function
def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size):
    # build the model
    NN_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(11,)),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    # compile the model
    NN_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(lr), metrics=['accuracy'])
    
    # train the model
    history = NN_model.fit(
        X_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.2, verbose=0
    )
    
    return NN_model, history

# scale the data
scaler_NN = MinMaxScaler()
X_train_scaled = scaler_NN.fit_transform(X_train_NN)
X_val_scaled = scaler_NN.transform(X_val_NN)
X_test_scaled = scaler_NN.transform(X_test)

# training Loop
least_val_loss = float('inf')
least_loss_model = None
best_params = None
for num_nodes in [8, 16, 32]:
    for dropout_prob in [0, 0.2]:
        for lr in [0.01, 0.005, 0.001]:
            for batch_size in [32, 64, 128]:
                print(f"{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}")
                model, history = train_model(X_train_scaled, y_train_NN_one_hot, num_nodes, dropout_prob, lr, batch_size)
                val_loss, _ = model.evaluate(X_val_scaled, y_val_NN_one_hot)
                if val_loss < least_val_loss:
                    least_val_loss = val_loss
                    least_loss_model = model
                    best_params = (num_nodes, dropout_prob, lr, batch_size)

# Parameter names
param_names = ['num_nodes', 'dropout_prob', 'lr', 'batch_size']

# Creating a dictionary by zipping the names with the values
best_params_dict = dict(zip(param_names, best_params))

# predicting on the test set
y_pred_probs = least_loss_model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)

# Convert one-hot encoded y_test to label encoded for comparison
y_test_NN_label_encoded = np.argmax(y_test_NN_one_hot, axis=1)

# stop the timer
end_time_NN = time.time()

# calculate the execution time
duration_NN = end_time_NN - start_time_NN

# Print classification report
report = classification_report(y_test_NN_label_encoded, y_pred, output_dict=True)

# convert the report to a DataFrame
df_report = pd.DataFrame(report).transpose()

# add the column with the cell types
df_report.reset_index(inplace=True)
df_report.rename(columns={'index': 'Cell Type'}, inplace=True)

write_model_info_to_file(file_path, "Neural Network", duration_NN, best_params_dict, df_report)

# save the best models (check whether the file alleady exists)
if not os.path.exists(os.path.join(models_directory, 'best_model_LR.joblib')):
    dump(grid_search_LR, os.path.join(models_directory, 'best_model_LR.joblib'))
if not os.path.exists(os.path.join(models_directory, 'best_model_kNN.joblib')):
    dump(grid_search_kNN, os.path.join(models_directory, 'best_model_kNN.joblib'))
if not os.path.exists(os.path.join(models_directory, 'best_model_NB.joblib')):
    dump(grid_search_NB, os.path.join(models_directory, 'best_model_NB.joblib'))
if not os.path.exists(os.path.join(models_directory, 'best_model_RF.joblib')):
    dump(grid_search_RF, os.path.join(models_directory, 'best_model_RF.joblib'))
if not os.path.exists(os.path.join(models_directory, 'best_model_NN.joblib')):
    least_loss_model.save(os.path.join(models_directory, 'best_model_NN.h5'))