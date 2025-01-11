from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import mlflow
import numpy as np
import argparse

PSI_THRESHOLD = 0.2
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

BASELINE_DATA_PATH = "/opt/spark-data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PRODUCTION_DATA_PATH = "/opt/spark-data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

parser = argparse.ArgumentParser(description="A script with flags.")
parser.add_argument("--run-id", type=str, default="")
args = parser.parse_args()

run_id = args.run_id


def calculate_psi(base_df, prod_df, column):
    base_hist = base_df.select(column).rdd.flatMap(lambda x: x).histogram(10)
    prod_hist = prod_df.select(column).rdd.flatMap(lambda x: x).histogram(base_hist[0])

    base_distribution = np.array(base_hist[1]) / sum(base_hist[1])
    prod_distribution = np.array(prod_hist[1]) / sum(prod_hist[1])

    psi_values = (prod_distribution - base_distribution) * np.log(
        (prod_distribution + 1e-6) / (base_distribution + 1e-6)
    )
    return float(np.sum(psi_values))


def preprocess_data(spark, file_path):
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    for col in NUMERIC_COLUMNS:
        df = df.withColumn(col, F.col(col).cast(DoubleType()))
        avg_value = df.select(F.avg(col)).first()[0]
        df = df.fillna({col: avg_value})
    return df


def detect_drift():
    spark = (
        SparkSession.builder.appName("DriftDetection")
        .config("spark.jars.packages", "org.mlflow:mlflow-spark:2.19.0")
        .getOrCreate()
    )

    baseline_df = preprocess_data(spark, BASELINE_DATA_PATH)
    production_df = preprocess_data(spark, PRODUCTION_DATA_PATH)

    psi_scores = {}
    for col in NUMERIC_COLUMNS:
        psi_scores[col] = calculate_psi(baseline_df, production_df, col)

    with mlflow.start_run(run_id=run_id):
        drift_detected = any(score > PSI_THRESHOLD for score in psi_scores.values())
        mlflow.log_metric("drift_detected", int(drift_detected))

        for col, score in psi_scores.items():
            mlflow.log_metric(f"psi_{col}", score)


if __name__ == "__main__":
    detect_drift()
