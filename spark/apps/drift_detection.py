from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import mlflow
import argparse, os, boto3, math
from urllib.parse import urlparse
from pathlib import Path

PSI_THRESHOLD = 0.2
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
FEATURE_COLUMNS = ["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"]
TRAIN_DATA_URI = "s3://minio/data/train-data.csv"

parser = argparse.ArgumentParser(description="A script with flags.")
parser.add_argument("--run-id", type=str, default="")
parser.add_argument("--prod-uri", type=str, default="")
args = parser.parse_args()

run_id = args.run_id
prod_uri = args.prod_uri

def calculate_psi(spark: SparkSession,base_df: DataFrame, prod_df, column):
    base_hist = base_df.select(column).rdd.flatMap(lambda x: x).histogram(10)
    prod_hist = prod_df.select(column).rdd.flatMap(lambda x: x).histogram(base_hist[0])

    base_distribution = spark.sparkContext.parallelize(base_hist[1]).map(lambda x: x / sum(base_hist[1]))
    prod_distribution = spark.sparkContext.parallelize(prod_hist[1]).map(lambda x: x / sum(prod_hist[1]))

    psi_values = base_distribution.zip(prod_distribution).map(
        lambda x: (x[1] - x[0]) * math.log((x[1] + 1e-6) / (x[0] + 1e-6))
    )

    return psi_values.sum()


def preprocess_data(spark: SparkSession, run_id: str, file_path: str):
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url="http://minio:9000"
    )
    parsed = urlparse(file_path)
    bucket_name = parsed.netloc
    object_key = parsed.path.lstrip("/")
    local_path = os.path.join(f'/opt/spark-data/{run_id}/data-drift', object_key)

    Path(os.path.dirname(local_path)).mkdir(parents=True, exist_ok=True)

    with open(local_path, 'wb') as f:
        s3.download_fileobj(bucket_name, object_key, f)

    df = spark.read.csv(local_path, header=True, inferSchema=True)

    df = df.select(*FEATURE_COLUMNS)

    for col in NUMERIC_COLUMNS:
        df = df.withColumn(col, F.col(col).cast(DoubleType()))
        avg_value = df.select(F.avg(col)).first()[0]
        df = df.fillna({col: avg_value})

    return df


def detect_drift():
    spark: SparkSession = SparkSession.builder.appName("DriftDetection") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.5.4") \
        .getOrCreate()

    baseline_df = preprocess_data(spark, run_id, TRAIN_DATA_URI)
    production_df = preprocess_data(spark, run_id, prod_uri)

    psi_scores = {}
    for col in NUMERIC_COLUMNS:
        psi_scores[col] = calculate_psi(spark, baseline_df, production_df, col)

    with mlflow.start_run(run_id=run_id):
        drift_detected = any(score > PSI_THRESHOLD for score in psi_scores.values())
        mlflow.log_metric("drift_detected", int(drift_detected))

        for col, score in psi_scores.items():
            mlflow.log_metric(f"psi_{col}", score)

    directory_path = Path(f'/opt/spark-data/{run_id}')

    for root, dirs, files in os.walk(directory_path, topdown=False):
        for file in files:
            os.unlink(Path(root) / file)
        for dir in dirs:
            os.rmdir(Path(root) / dir)

    directory_path.rmdir()


if __name__ == "__main__":
    detect_drift()
