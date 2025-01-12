from pyspark.sql import DataFrame, SparkSession
import argparse, os, boto3
from urllib.parse import urlparse
from pathlib import Path

ALL_DATA_URI = "s3://minio/data/all_data.csv"
ALL_COLUMNS = ["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges", "Churn"]

parser = argparse.ArgumentParser(description="A script with flags.")
parser.add_argument("--run-id", type=str, default="")
parser.add_argument("--prod-uri", type=str, default="")
args = parser.parse_args()

run_id = args.run_id
prod_uri = args.prod_uri

def load_df(spark: SparkSession, run_id: str, file_uri: str):
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url="http://minio:9000"
    )
    parsed = urlparse(file_uri)
    bucket_name = parsed.netloc
    object_key = parsed.path.lstrip("/")
    local_path = os.path.join(f'/opt/spark-data/{run_id}', object_key)

    Path(os.path.dirname(local_path)).mkdir(parents=True, exist_ok=True)

    with open(local_path, 'wb') as f:
        s3.download_fileobj(bucket_name, object_key, f)

    df = spark.read.csv(local_path, header=True, inferSchema=True)

    for col in ALL_COLUMNS:
        if col not in df.columns:
            raise RuntimeError("Feature columns not complete")

    df = df.select(*ALL_COLUMNS)

    return df

def upload_data(df: DataFrame, run_id: str, file_uri: str):
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url="http://minio:9000"
    )
    local_path = os.path.join(f'/opt/spark-data/{run_id}', "combined_data.csv")

    df.write.csv(local_path, header=True, mode="overwrite")

    parsed = urlparse(file_uri)
    bucket_name = parsed.netloc
    object_key = parsed.path.lstrip("/")

    s3.upload_file(local_path, bucket_name, object_key)

def combine_data():
    spark: SparkSession = SparkSession.builder.appName("CombineData") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.5.4") \
        .getOrCreate()

    baseline_df = load_df(spark, run_id, ALL_DATA_URI)
    production_df = load_df(spark, run_id, prod_uri)

    combined_df = baseline_df.union(production_df)
    upload_data(combined_df, run_id, ALL_DATA_URI)

    directory_path = Path(f'/opt/spark-data/{run_id}')

    for root, dirs, files in os.walk(directory_path, topdown=False):
        for file in files:
            os.unlink(Path(root) / file)
        for dir in dirs:
            os.rmdir(Path(root) / dir)

    directory_path.rmdir()


if __name__ == "__main__":
    combine_data()
