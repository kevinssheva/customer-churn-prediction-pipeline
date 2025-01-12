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

    @task.branch()
    def decide_retrain(data_drift_run_id: str):
        client = MlflowClient()
        run = client.get_run(data_drift_run_id)

        if run.data.metrics["drift_detected"] is None:
            raise RuntimeError("drift_detected metric not found")

        if run.data.metrics["drift_detected"] == 0:
            return "end_run"
        else:
            return "start_retrain"

    ddd_experiment_id = create_or_get_experiment(experiment_name=DATA_DRIFT_EXPERIMENT_NAME)
    ddd_run_id = create_run(ddd_experiment_id)

    detect_drift_task = SparkSubmitOperator(
        task_id="detect_drift",
        application="/opt/spark-apps/drift_detection.py",
        application_args=[
            "--run-id", ddd_run_id,
            "--prod-uri", "s3://sample/prod-split-1.csv",
        ],
        conn_id=SPARK_CONN_ID,
    )

    decide_retrain_task = decide_retrain(ddd_run_id)

    combine_untrained_data_task = SparkSubmitOperator(
        task_id="combine_untrained_data",
        application="/opt/spark-apps/combine_data.py",
        application_args=[
            "--run-id", ddd_run_id,
            "--data-a-uri", "s3://data/untrained-data.csv",
            "--data-b-uri", "s3://sample/prod-split-1.csv",
            "--dest-uri", "s3://data/untrained-data.csv",
        ],
        conn_id=SPARK_CONN_ID,
    )

    detect_drift_task >> combine_untrained_data_task >> decide_retrain_task

    end_run = EmptyOperator(task_id="end_run")
    start_retrain = EmptyOperator(task_id="start_retrain")

    retrain_experiment_id = create_or_get_experiment(experiment_name=TRAIN_MODEL_EXPERIMENT_NAME)
    retrain_run_id = create_run(retrain_experiment_id)

    combine_all_data_task = SparkSubmitOperator(
        task_id="combine_all_data",
        application="/opt/spark-apps/combine_data.py",
        application_args=[
            "--run-id", retrain_run_id,
            "--data-a-uri", "s3://data/train-data.csv",
            "--data-b-uri", "s3://data/untrained-data.csv",
            "--dest-uri", "s3://data/train-data.csv",
        ],
        conn_id=SPARK_CONN_ID,
    )

    retrain_data_task = SparkSubmitOperator(
        task_id='retrain_data',
        application='/opt/spark-apps/train_model.py',
        application_args=[
            '--run-id', retrain_run_id,
        ],
    )

    decide_retrain_task >> end_run
    decide_retrain_task >> start_retrain >> retrain_experiment_id >> retrain_run_id >> combine_all_data_task >> retrain_data_task >> end_run

consume_prod_data()
