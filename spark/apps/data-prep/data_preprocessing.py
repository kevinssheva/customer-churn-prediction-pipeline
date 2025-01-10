from pandas.core.series import CombinedDatetimelikeProperties
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark
import sys, os

numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
FILE_PATH = "/opt/spark-data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

combined_data = sys.argv[1]
run_id = sys.argv[2]

def process_data() -> None:
    spark = SparkSession.builder \
        .appName("DataProcessing") \
        .config("spark.jars.packages", "org.mlflow:mlflow-spark:2.19.0") \
        .config('spark.hadoop.fs.s3a.endpoint', 'http://localhost:9001') \
        .config('spark.hadoop.fs.s3a.access.key', os.getenv('AWS_ACCESS_KEY_ID')) \
        .config('spark.hadoop.fs.s3a.secret.key', os.getenv('AWS_SECRET_ACCESS_KEY')) \
        .config('spark.hadoop.fs.s3a.path.style.access', 'true') \
        .config('spark.hadoop.fs.s3a.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem') \
        .getOrCreate()

    if combined_data == "":
        df = spark.read.csv(FILE_PATH, header=True, inferSchema=True)
    else:
        df = spark.read.parquet(f"s3://mlflow/{combined_data}", header=True, inferSchema=True)

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
        model = pipeline.fit(df)
        processed_df = model.transform(df)

        mlflow.spark.log_model(model, "preprocessing_pipeline")

        numeric_stats = processed_df.select([
            F.mean(F.col("scaled_numeric_features")).alias("mean_scaled_features"),
            F.stddev(F.col("scaled_numeric_features")).alias("std_scaled_features")
        ]).first()

        mlflow.log_metrics({
            "mean_scaled_features": float(numeric_stats["mean_scaled_features"]),
            "std_scaled_features": float(numeric_stats["std_scaled_features"])
        })
        
        final_columns = [f"{col}_indexed" for col in categorical_columns]
        final_columns.append("scaled_numeric_features")

        dataset = mlflow.data.from_spark(processed_df)
        mlflow.log_input(dataset, context="training")

if __name__ == "__main__":
    processed_df = process_data()
