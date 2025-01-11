from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess


def run_mlflow_script(script_path):
    """Runs an MLflow script."""
    try:
        subprocess.check_call(['python', script_path])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to run MLflow script: {script_path}. Error: {e}")


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='churn_prediction_pipeline',
    default_args=default_args,
    description='End-to-end churn prediction pipeline',
    schedule_interval='@weekly',
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
    start = DummyOperator(task_id='start')

    data_preprocessing = SparkSubmitOperator(
        task_id='data_preprocessing',
        application='/opt/spark-apps/data_preprocessing.py',
        conn_id='spark_default',  # Airflow connection ID for Spark
        name='data_preprocessing',
        application_args=[],
        verbose=True
    )

    feature_engineering = SparkSubmitOperator(
        task_id='feature_engineering',
        application='/opt/spark-apps/feature_engineering.py',
        conn_id='spark_default',
        name='feature_engineering',
        application_args=[],
        verbose=True
    )

    train_model = SparkSubmitOperator(
        task_id='train_model',
        application='/opt/spark-apps/train_model.py',
        conn_id='spark_default',
        name='train_model',
        application_args=[],
        verbose=True
    )

    evaluate_model = SparkSubmitOperator(
        task_id='evaluate_model',
        application='/opt/spark-apps/evaluate_model.py',
        conn_id='spark_default',
        name='evaluate_model',
        application_args=[],
        verbose=True
    )

    track_model = PythonOperator(
        task_id='track_model',
        python_callable=run_mlflow_script,
        op_args=['mlflow/model_tracking.py']
    )

    end = DummyOperator(task_id='end')

    # DAG Workflow
    start >> data_preprocessing >> feature_engineering >> train_model >> evaluate_model >> track_model >> end
