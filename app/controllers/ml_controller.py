# ml_controller.py
from flask import Blueprint, request, jsonify
from sklearn.preprocessing import StandardScaler
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models.model import load_data, clean_data, split_data, select_features, train_model
import pandas as pd

ML_router = Blueprint("ML_router", __name__)

# Load Tensorflow model
MODEL_PATH = '/app/best_model2.h5'
model = keras.models.load_model(MODEL_PATH)

# Load Scaler
scaler = StandardScaler()
SCALER_PATH = '/app/scaler2.pkl'
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)


def preprocess_input_data(input_data):
    relevant_features = ['EK', 'Skewness']
    # Convertir el diccionario de entrada a un DataFrame
    input_df = pd.DataFrame(input_data)
    processed_data = input_df[relevant_features]
    processed_data_scaled = scaler.transform(processed_data)
    return processed_data_scaled


@ML_router.route("/ml", methods=["POST"])
def create_pred():
    pred_input = request.get_json()
    if not pred_input:
        return jsonify({"message": "Invalid input"}), 400

    # Obtén los datos de entrada desde el cuerpo de la solicitud
    input_data = pred_input.get("data")
    if not input_data:
        return jsonify({"message": "Data is required"}), 400

    # Preprocesa los datos de entrada
    processed_data = preprocess_input_data(input_data)

    # Realiza la predicción
    prediction = model.predict(processed_data)

    # Formatea la respuesta y devuelve la predicción
    prediction_result = {"prediction": prediction.tolist()}
    return jsonify(prediction_result), 201
