from airflow import Dataset
from airflow.decorators import dag, task, task_group
from pendulum import datetime
from astro import sql as aql
from astro.files import File
from astro.dataframes.pandas import DataFrame
from airflow.operators.empty import EmptyOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from mlflow_provider.hooks.client import MLflowClientHook
import os
import pandas as pd

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

# Data parameters
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


@dag(
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
)
def data_preparation():
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(
        task_id="end",
        outlets=[Dataset("s3://" + DATA_BUCKET_NAME + "_" + PROCESSED_TRAIN_DATA_PATH)],
    )

    create_buckets_if_not_exists = S3CreateBucketOperator.partial(
        task_id="create_buckets_if_not_exists",
        aws_conn_id=AWS_CONN_ID,
    ).expand(bucket_name=[DATA_BUCKET_NAME, MLFLOW_ARTIFACT_BUCKET, XCOM_BUCKET])

    @task_group
    def prepare_mlflow_experiment():
        @task
        def list_existing_experiments(max_results=1000):
            "Get information about existing MLFlow experiments."

            mlflow_hook = MLflowClientHook(mlflow_conn_id=MLFLOW_CONN_ID)
            existing_experiments_information = mlflow_hook.run(
                endpoint="api/2.0/mlflow/experiments/search",
                request_params={"max_results": max_results},
            ).json()

            return existing_experiments_information

        @task.branch
        def check_if_experiment_exists(
            experiment_name, existing_experiments_information
        ):
            "Check if the specified experiment already exists."

            if existing_experiments_information:
                existing_experiment_names = [
                    experiment["name"]
                    for experiment in existing_experiments_information["experiments"]
                ]
                if experiment_name in existing_experiment_names:
                    return "prepare_mlflow_experiment.experiment_exists"
                else:
                    return "prepare_mlflow_experiment.create_experiment"
            else:
                return "prepare_mlflow_experiment.create_experiment"

        @task
        def create_experiment(experiment_name, artifact_bucket):
            """Create a new MLFlow experiment with a specified name.
            Save artifacts to the specified S3 bucket."""

            mlflow_hook = MLflowClientHook(mlflow_conn_id=MLFLOW_CONN_ID)
            new_experiment_information = mlflow_hook.run(
                endpoint="api/2.0/mlflow/experiments/create",
                request_params={
                    "name": experiment_name,
                    "artifact_location": f"s3://{artifact_bucket}/",
                },
            ).json()

            return new_experiment_information

        experiment_already_exists = EmptyOperator(task_id="experiment_exists")

        @task(
            trigger_rule="none_failed",
        )
        def get_current_experiment_id(experiment_name, max_results=1000):
            "Get the ID of the specified MLFlow experiment."

            mlflow_hook = MLflowClientHook(mlflow_conn_id=MLFLOW_CONN_ID)
            experiments_information = mlflow_hook.run(
                endpoint="api/2.0/mlflow/experiments/search",
                request_params={"max_results": max_results},
            ).json()

            for experiment in experiments_information["experiments"]:
                if experiment["name"] == experiment_name:
                    return experiment["experiment_id"]

            raise ValueError(f"{experiment_name} not found in MLFlow experiments.")

        experiment_id = get_current_experiment_id(
            experiment_name=EXPERIMENT_NAME,
            max_results=MAX_RESULTS_MLFLOW_LIST_EXPERIMENTS,
        )

        (
            check_if_experiment_exists(
                experiment_name=EXPERIMENT_NAME,
                existing_experiments_information=list_existing_experiments(
                    max_results=MAX_RESULTS_MLFLOW_LIST_EXPERIMENTS
                ),
            )
            >> [
                experiment_already_exists,
                create_experiment(
                    experiment_name=EXPERIMENT_NAME,
                    artifact_bucket=MLFLOW_ARTIFACT_BUCKET,
                ),
            ]
            >> experiment_id
        )

    process_data = SparkSubmitOperator(
        task_id='process_data',
        application='/spark/apps/data-prep/data_preprocessing.py',
        conn_id=SPARK_CONN_ID,
        application_args=[
            '--input_path', f"spark/data/{FILE_PATH}",
            '--experiment_id', "{{ task_instance.xcom_pull(task_ids='prepare_mlflow_experiment.get_current_experiment_id') }}",
            '--numeric_columns', ','.join(NUMERIC_COLUMNS),
            '--output_path', os.path.join("s3://", DATA_BUCKET_NAME, PROCESSED_TRAIN_DATA_PATH)
        ],
        conf={
            'spark.jars.packages': 'org.mlflow:mlflow-spark:2.19.0',
            'spark.hadoop.fs.s3a.aws.credentials.provider': 'org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider'
        }
    )

    # Task dependencies
    (
        start
        >> create_buckets_if_not_exists
        >> prepare_mlflow_experiment()
        >> process_data
        >> end
    )


data_preparation()
