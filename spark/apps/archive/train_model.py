from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark

spark = SparkSession.builder.appName('TrainModel').getOrCreate()

# Load data
data = spark.read.parquet('path/to/cleaned_data.parquet')

# Feature engineering
assembler = VectorAssembler(inputCols=['tenure', 'MonthlyCharges', 'TotalCharges'], outputCol='features')
lr = LogisticRegression(featuresCol='features', labelCol='Churn')

pipeline = Pipeline(stages=[assembler, lr])
model = pipeline.fit(data)

# Log model to MLflow
mlflow.set_experiment('telco_churn')
with mlflow.start_run():
    mlflow.spark.log_model(model, 'churn_model')
