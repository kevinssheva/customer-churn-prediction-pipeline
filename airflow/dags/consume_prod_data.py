from airflow.decorators import dag, task
import os
from mlflow.entities import Experiment, Run
from pendulum import datetime
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from mlflow import MlflowClient
import mlflow
import logging

# Spark parameters
SPARK_CONN_ID = "spark_default"
MLFLOW_URI = "http://mlflow-server:5000"

TRAIN_MODEL_EXPERIMENT_NAME = "TRAIN_MODEL"
DATA_DRIFT_EXPERIMENT_NAME = "DATA_DRIFT_DETECTION"

@dag(
    schedule="@daily",
    start_date=datetime(2023, 1, 1),
    catchup=False,
)
def consume_prod_data():
    mlflow.set_tracking_uri(MLFLOW_URI)

    @task
    def get_last_train_experiment_run(experiment_name) -> str:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            raise RuntimeError("Last Train Experiment not Fonud")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )

        if len(runs) == 0:
            raise RuntimeError("Train Experiment has no runs")

        return runs[0].info.run_id

    @task
    def get_last_train_experiment_data(train_run_id):
        client = MlflowClient()
        artifacts = client.list_artifacts(train_run_id, path="data/processed_dataset")

        artifact_exists = any(artifact.path == "data/processed_dataset/processed_dataset.csv" for artifact in artifacts)

        if not artifact_exists:
            raise RuntimeError("Artifact does not exist")

        return f"s3://mlflow/{train_run_id}/artifacts/data/processed_dataset/processed_dataset.csv"

    train_run_id = get_last_train_experiment_run(TRAIN_MODEL_EXPERIMENT_NAME)
    train_data_uri = get_last_train_experiment_data(train_run_id=train_run_id)

    SparkSubmitOperator(
        task_id='calculate_psi',
        application='/opt/spark-apps/calculate_psi.py',
        application_args=[
            '--train-data-uri', train_data_uri,
        ],
    )

consume_prod_data()
