from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, lag
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import pandas as pd
import pickle

# Initialiser une session Spark
spark = SparkSession.builder.appName("Netflix Stock Price Prediction").getOrCreate()

# Charger le fichier CSV
data_path = "/content/stock_data.csv"
data = spark.read.csv(data_path, header=True, inferSchema=True)

# Afficher un aperçu des données
data.show(5)
data.printSchema()

# Prétraitement des données
# Convertir les colonnes en types numériques et gérer les données manquantes
columns = ['Date', 'Open', 'High', 'Low', 'Volume', 'Close']
data = data.select([col(c).alias(c) for c in columns])
data = data.withColumn("Date", unix_timestamp(col("Date"), "yyyy-MM-dd").cast("timestamp"))
data = data.dropna()

# Créer des colonnes de décalage pour les séries temporelles
window_spec = Window.orderBy("Date")
data = data.withColumn("Prev_Close", lag("Close", 1).over(window_spec))
data = data.dropna()

# Configurer le vecteur de caractéristiques
assembler = VectorAssembler(inputCols=["Prev_Close"], outputCol="features")
data = assembler.transform(data)

# Diviser les données en ensembles d'entraînement et de test
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Entraîner un modèle de régression linéaire
lr = LinearRegression(featuresCol="features", labelCol="Close")
model = lr.fit(train_data)

# Évaluer le modèle
summary = model.evaluate(test_data)
print("R2:", summary.r2)
print("RMSE:", summary.rootMeanSquaredError)

# Enregistrer le modèle en tant que fichier pickle
# Convertir le modèle Spark ML en un modèle scikit-learn si nécessaire (via pandas DataFrame)
pandas_data = data.select("features", "Close").toPandas()
X = list(pandas_data["features"])
y = pandas_data["Close"]

# Entraîner un modèle scikit-learn
from sklearn.linear_model import LinearRegression as SklearnLR
sklearn_model = SklearnLR().fit(X, y)

# Sauvegarder le modèle scikit-learn
with open("stock_model.pkl", "wb") as f:
    pickle.dump(sklearn_model, f)

print("Modèle enregistré sous 'stock_model.pkl'")
