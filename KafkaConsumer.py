import os
import time
import csv
import pickle
from kafka import KafkaConsumer
from google.cloud import bigquery
from google.oauth2 import service_account

# Chemin du modèle pickle
MODEL_PATH = "trained_model.pkl"

# Chargement du modèle pickle
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

# BigQuery configuration
credentials = service_account.Credentials.from_service_account_file("real-time-stock-prediction.json")
bq_client = bigquery.Client(credentials=credentials)

# Table BigQuery
table_id = "real-time-stock-prediction.Stocks_data.Netflix_Stocks"

# Kafka Consumer
consumer = KafkaConsumer("Stock_Topic", bootstrap_servers="localhost:9092")

# Accumuler les données
accumulated_data = []

# Batch size et intervalle
BATCH_SIZE = 100
TIME_INTERVAL = 60  # 60 secondes

# Préparer les données pour le modèle
def prepare_features(data):
    """
    Prépare une seule feature pour le modèle, comme attendu par le modèle LinearRegression.
    Ici, nous utilisons 'Close' comme seule feature.
    """
    return [[data["Close"]]]  # Feature attendue sous forme de liste de listes

# Fonction pour écrire les données dans un fichier CSV
def write_to_csv(data, file_path):
    if not data:
        return
    with open(file_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

# Fonction pour insérer dans BigQuery
def insert_into_bigquery(file_path, table_id):
    with open(file_path, "rb") as file:
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            autodetect=True
        )
        load_job = bq_client.load_table_from_file(file, table_id, job_config=job_config)
        load_job.result()

# Fonction principale pour traiter les messages Kafka
def main():
    batch_file_path = "predicted_batch_data.csv"
    last_insert_time = time.time()

    try:
        for msg in consumer:
            # Décoder le message Kafka
            data = eval(msg.value.decode("utf-8"))
            print(f"Received data: {data}")

            # Préparer les features et effectuer la prédiction
            try:
                features = prepare_features(data)
                predicted_close = model.predict(features)[0]  # Prédiction
                data["PredictedClose"] = float(predicted_close)
                print(f"Predicted Close: {predicted_close}")
            except Exception as e:
                print(f"Prediction error: {e}")
                continue

            # Ajouter aux données accumulées
            accumulated_data.append(data)

            # Si le batch est complet ou si le temps est écoulé, insérer dans BigQuery
            if len(accumulated_data) >= BATCH_SIZE or time.time() - last_insert_time > TIME_INTERVAL:
                write_to_csv(accumulated_data, batch_file_path)
                insert_into_bigquery(batch_file_path, table_id)
                accumulated_data.clear()
                last_insert_time = time.time()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        consumer.close()
        if accumulated_data:
            write_to_csv(accumulated_data, batch_file_path)
            insert_into_bigquery(batch_file_path, table_id)

if __name__ == "__main__":
    main()
