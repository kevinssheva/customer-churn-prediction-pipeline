from airflow.decorators import dag, task
import os
from mlflow.entities import Experiment, Run
from pendulum import datetime
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from mlflow import MlflowClient
import mlflow
import logging

SPARK_CONN_ID = "spark_default"
TRAIN_MODEL_EXPERIMENT_NAME = "TRAIN_MODEL"
MLFLOW_URI = "http://mlflow-server:5000"

@dag(
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
)
def train_model():
    mlflow.set_tracking_uri(MLFLOW_URI)

    @task
    def create_or_get_experiment(experiment_name) -> str:
        """Create a new MLFlow experiment with a specified name.
        Save artifacts to the specified S3 bucket."""

        mlflow_client = MlflowClient()
        experiment = mlflow_client.get_experiment_by_name(experiment_name)

        if experiment is None:
            experiment_id = mlflow_client.create_experiment(
                name=experiment_name,
                artifact_location=f"s3://mlflow/",
            )

            experiment = mlflow_client.get_experiment(experiment_id)


        return experiment.experiment_id

    @task
    def create_run(experiment_id: str) -> str:
        mlflow_client = MlflowClient()
        run = mlflow_client.create_run(experiment_id)

        logging.info(f"experiment_id: {experiment_id}")
        return run.info.run_id

    experiment_id = create_or_get_experiment(experiment_name=TRAIN_MODEL_EXPERIMENT_NAME)

    run_id = create_run(experiment_id)

    SparkSubmitOperator(
        task_id='train_original_data',
        application='/opt/spark-apps/train_model.py',
        application_args=[
            '--run-id', run_id,
        ],
    )


train_model()
