from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark
import mlflow.data
from mlflow.models import infer_signature
import os, argparse, boto3
from urllib.parse import urlparse
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pathlib import Path

numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
TRAIN_DATA_URI = "s3://data/train-data.csv"

parser = argparse.ArgumentParser(description="A script with flags.")
parser.add_argument("--run-id", type=str, default="")
args = parser.parse_args()

run_id = args.run_id

def train_model() -> None:
    spark = SparkSession.builder \
        .appName("TrainModelOriginalData") \
        .getOrCreate()

    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url="http://minio:9000"
    )
    parsed = urlparse(TRAIN_DATA_URI)
    bucket_name = parsed.netloc
    object_key = parsed.path.lstrip("/")
    local_path = os.path.join(f'/opt/spark-data/{run_id}/data-drift', object_key)

    Path(os.path.dirname(local_path)).mkdir(parents=True, exist_ok=True)

    with open(local_path, 'wb') as f:
        s3.download_fileobj(bucket_name, object_key, f)

    df = spark.read.csv(local_path, header=True, inferSchema=True)

    df = df.drop("customerID")
    df = df.withColumn("TotalCharges", F.col("TotalCharges").cast(DoubleType()))
    avg_total_charges = df.select(F.avg("TotalCharges")).first()[0]
    df = df.fillna(avg_total_charges, subset=["TotalCharges"])
    df = df.filter(F.col("tenure") != 0)
    df = df.withColumn("SeniorCitizen", 
                      F.when(F.col("SeniorCitizen") == 1, "Yes")
                       .otherwise("No"))
    
    categorical_columns = [field.name for field in df.schema.fields 
                         if field.dataType.simpleString() == "string"]

    stages = []
    for column in categorical_columns:
        indexer = StringIndexer(
            inputCol=column,
            outputCol=f"{column}_indexed",
            handleInvalid="keep"
        )
        stages.append(indexer)
    
    numeric_assembler = VectorAssembler(
        inputCols=numeric_columns,
        outputCol="numeric_features",
        handleInvalid="keep"
    )
    stages.append(numeric_assembler)
    
    scaler = StandardScaler(
        inputCol="numeric_features",
        outputCol="scaled_numeric_features",
        withStd=True,
        withMean=True
    )
    stages.append(scaler)
    
    pipeline = Pipeline(stages=stages)
    
    with mlflow.start_run(run_id=run_id):
        preprocessing = pipeline.fit(df)
        processed_df = preprocessing.transform(df)

        final_columns = [f"{col}_indexed" for col in categorical_columns] + ["scaled_numeric_features"]

        assembler = VectorAssembler(
            inputCols=final_columns,
            outputCol="features"
        )
        processed_df = assembler.transform(processed_df)

        label_indexer = StringIndexer(
            inputCol="Churn",
            outputCol="label"
        )
        processed_df = label_indexer.fit(processed_df).transform(processed_df)

        dataset = mlflow.data.from_spark(processed_df)
        mlflow.log_input(dataset, context="training")

        blor = LogisticRegression(maxIter=2)
        model = blor.fit(processed_df)

        features_df = processed_df.select("features")
        prediction_df = model.transform(processed_df)

        signature = infer_signature(features_df, prediction_df.select("prediction"))

        mlflow.spark.log_model(
            model,
            "model",
            signature=signature,
        )

        metrics = {}
        evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        metrics["accuracy"] = evaluator_accuracy.evaluate(prediction_df)
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
        metrics["f1_score"] = evaluator_f1.evaluate(prediction_df)
        evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
        metrics["precision"] = evaluator_precision.evaluate(prediction_df)
        evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
        metrics["recall"] = evaluator_recall.evaluate(prediction_df)

        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

    directory_path = Path(f'/opt/spark-data/{run_id}')

    for root, dirs, files in os.walk(directory_path, topdown=False):
        for file in files:
            os.unlink(Path(root) / file)
        for dir in dirs:
            os.rmdir(Path(root) / dir)

    directory_path.rmdir()


if __name__ == "__main__":
    processed_df = train_model()
