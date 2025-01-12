from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
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

        uri = f"s3://mlflow/{train_run_id}/artifacts/data/processed_dataset/processed_dataset.csv"
        logging.info(f"uri {uri}")
        return uri

    @task.branch()
    def decide_retrain(data_drift_run_id: str):
        client = MlflowClient()
        run = client.get_run(data_drift_run_id)

        if run.data.metrics["drift_detected"] is None:
            raise RuntimeError("drift_detected metric not found")

        if run.data.metrics["drift_detected"] == 0:
            return "end_run"
        else:
            return "retrain_data"

    experiment_id = create_or_get_experiment(experiment_name=DATA_DRIFT_EXPERIMENT_NAME)
    run_id = create_run(experiment_id)

    train_run_id = get_last_train_experiment_run(TRAIN_MODEL_EXPERIMENT_NAME)
    train_data_uri = get_last_train_experiment_data(train_run_id=train_run_id)

    detect_drift_task = SparkSubmitOperator(
        task_id="detect_drift",
        application="/opt/spark-apps/drift_detection.py",
        application_args=[
            "--run-id", run_id,
            "--baseline-uri", train_data_uri,
            "--prod-uri", "s3://data/prod-split.csv",
        ],
        conn_id=SPARK_CONN_ID,
    )

    decide_retrain_task = decide_retrain(run_id)

    end_run = EmptyOperator(task_id="end_run")

    combine_data_task = SparkSubmitOperator(
        task_id="combine_data",
        application="/opt/spark-apps/combine_data.py",
        application_args=[
            "--run-id", run_id,
            "--prod-uri", "s3://data/prod-split.csv",
        ],
        conn_id=SPARK_CONN_ID,
    )

    retrain_data_task = SparkSubmitOperator(
        task_id='retrain_data',
        application='/opt/spark-apps/train_model.py',
        application_args=[
            '--run-id', run_id,
        ],
    )

    detect_drift_task >> combine_data_task >> decide_retrain_task

    decide_retrain_task >> end_run
    decide_retrain_task >> retrain_data_task >> end_run

consume_prod_data()
