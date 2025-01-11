from airflow.decorators import dag, task
from mlflow.entities import Experiment, Run
from pendulum import datetime
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from mlflow import MlflowClient
import logging

# Spark parameters
SPARK_CONN_ID = "spark_default"

@dag(
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
)
def data_preparation():
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

        return run.info.run_id

    experiment_id = create_or_get_experiment(experiment_name="DATA_PREPROCESS")

    logging.info(f"experiment_id: {experiment_id}")
    run_id = create_run(experiment_id)

    SparkSubmitOperator(
        task_id='process_data',
        application='/opt/spark-apps/data-prep/data_preprocessing.py',
        application_args=[
            '--run-id', run_id,
        ],
    )


data_preparation()
