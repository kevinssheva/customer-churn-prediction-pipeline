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
import os, argparse
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
FILE_PATH = "/opt/spark-data/train-original-split.csv"

parser = argparse.ArgumentParser(description="A script with flags.")
parser.add_argument("--run-id", type=str, default="")
parser.add_argument("--combined-data", type=str, default="")
args = parser.parse_args()

run_id = args.run_id
combined_data = args.combined_data

def train_model() -> None:
    spark = SparkSession.builder \
        .appName("TrainModelOriginalData") \
        .config("spark.jars.packages", "org.mlflow:mlflow-spark:2.19.0") \
        .config('spark.hadoop.fs.s3a.endpoint', 'http://localhost:9000') \
        .config('spark.hadoop.fs.s3a.access.key', os.getenv('AWS_ACCESS_KEY_ID')) \
        .config('spark.hadoop.fs.s3a.secret.key', os.getenv('AWS_SECRET_ACCESS_KEY')) \
        .config('spark.hadoop.fs.s3a.path.style.access', 'true') \
        .config('spark.hadoop.fs.s3a.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem') \
        .getOrCreate()

    if combined_data == "":
        df = spark.read.csv(FILE_PATH, header=True, inferSchema=True)
    else:
        df = spark.read.csv(f"s3://mlflow/{combined_data}", header=True, inferSchema=True)

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

        output_path = "/tmp/processed_dataset.csv"
        processed_df.toPandas().to_csv(output_path, index=False)

        mlflow.log_artifact(output_path, artifact_path="data/processed_dataset")
        os.remove(output_path)

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


if __name__ == "__main__":
    processed_df = train_model()
