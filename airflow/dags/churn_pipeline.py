from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from utils import run_spark_script, run_mlflow_script

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'churn_prediction_pipeline',
    default_args=default_args,
    description='End-to-end churn prediction pipeline',
    schedule_interval='@weekly',
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
    start = DummyOperator(task_id='start')

    data_preprocessing = PythonOperator(
        task_id='data_preprocessing',
        python_callable=run_spark_script,
        op_args=['spark/apps/data_preprocessing.py']
    )

    feature_engineering = PythonOperator(
        task_id='feature_engineering',
        python_callable=run_spark_script,
        op_args=['spark/apps/feature_engineering.py']
    )

    train_model = PythonOperator(
        task_id='train_model',
        python_callable=run_spark_script,
        op_args=['spark/apps/train_model.py']
    )

    evaluate_model = PythonOperator(
        task_id='evaluate_model',
        python_callable=run_spark_script,
        op_args=['spark/apps/evaluate_model.py']
    )

    track_model = PythonOperator(
        task_id='track_model',
        python_callable=run_mlflow_script,
        op_args=['mlflow/model_tracking.py']
    )

    end = DummyOperator(task_id='end')

    # DAG Workflow
    start >> data_preprocessing >> feature_engineering >> train_model >> evaluate_model >> track_model >> end
