from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from utils import simulate_drift, monitor_drift

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'drift_simulation_pipeline',
    default_args=default_args,
    description='Simulate and monitor drift',
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    simulate = PythonOperator(
        task_id='simulate_drift',
        python_callable=simulate_drift,
    )

    monitor = PythonOperator(
        task_id='monitor_drift',
        python_callable=monitor_drift,
    )

    simulate >> monitor
