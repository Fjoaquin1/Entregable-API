import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

file_path = "/Users/joaquincodega/Desktop/Gitrepo/Entega_Api/app/Pulsar_cleaned.csv"

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    cleaned_data = data.dropna()
    return cleaned_data

def split_data(data):
    X = data.drop('Class', axis=1)
    y = data['Class']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def select_features(data):
    relevant_features = ['EK', 'Skewness']
    X_selected = data[relevant_features]
    return X_selected

def train_model(X_train, y_train):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model