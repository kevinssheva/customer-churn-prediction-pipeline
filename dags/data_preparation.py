from airflow.decorators import dag, task, task_group
from mlflow.entities import Experiment
from mlflow.store.entities.paged_list import PagedList
from pendulum import datetime
from airflow.operators.empty import EmptyOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
import os
from mlflow import MlflowClient
import mlflow, mlflow.entities

# Constants used in the DAG
FILE_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_TRAIN_DATA_PATH = "train_data.csv"

# AWS S3 parameters
AWS_CONN_ID = "aws_default"
DATA_BUCKET_NAME = "data"
MLFLOW_ARTIFACT_BUCKET = "mlflowdatachurn"
XCOM_BUCKET = "localxcom"

# MLFlow parameters
MLFLOW_CONN_ID = "mlflow_default"
EXPERIMENT_NAME = "customer_churn"
MAX_RESULTS_MLFLOW_LIST_EXPERIMENTS = 100

# Spark parameters
SPARK_CONN_ID = "spark_default"

@dag(
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
)
def data_preparation():
    @task_group
    def prepare_mlflow_experiment():
        @task
        def get_run_id(**kwargs):
            run_id = kwargs['dag_run'].run_id
            return run_id

        @task
        def create_experiment(experiment_name, artifact_bucket):
            """Create a new MLFlow experiment with a specified name.
            Save artifacts to the specified S3 bucket."""

            mlflow_client = MlflowClient()
            mlflow_client.create_experiment(
                name=experiment_name,
                artifact_location=f"s3://{artifact_bucket}/",
            )

        dag_run_id = get_run_id()

        exp_run_id = create_experiment(experiment_name=dag_run_id, artifact_bucket=MLFLOW_ARTIFACT_BUCKET)

        return exp_run_id


    exp_run_id = prepare_mlflow_experiment()

    SparkSubmitOperator(
        task_id='process_data',
        application='/opt/spark-apps/data-prep/data_preprocessing.py',
        conn_id=SPARK_CONN_ID,
        application_args=[
            '--run-id', exp_run_id,
            '--combined-data', "",
        ],
    )


data_preparation()
