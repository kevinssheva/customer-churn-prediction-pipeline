from airflow.decorators import dag, task
from airflow.operators.python import BranchPythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.empty import EmptyOperator
from mlflow import MlflowClient
import mlflow
from pendulum import datetime
import logging

# Konfigurasi Spark dan MLflow
SPARK_CONN_ID = "spark_default"
MLFLOW_URI = "http://mlflow-server:5000"
PSI_THRESHOLD = 0.2


@dag(
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
)
def detect_drift():
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
    def check_drift_from_mlflow(experiment_name: str) -> str:
        client = MlflowClient()

        experiment = client.get_experiment_by_name(experiment_name)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"]
        )
        latest_run = runs[0]

        metrics = latest_run.data.metrics
        drift_detected = any(
            value > PSI_THRESHOLD
            for key, value in metrics.items()
            if key.startswith("psi_")
        )

        logging.info(f"Drift detected: {drift_detected}")
        return "retrain_model" if drift_detected else "skip_retraining"

    @task
    def create_run(experiment_id: str) -> str:
        mlflow_client = MlflowClient()
        run = mlflow_client.create_run(experiment_id)

        logging.info(f"experiment_id: {experiment_id}")
        return run.info.run_id

    experiment_id = create_or_get_experiment(experiment_name="DRIFT_DETECTION")

    run_id = create_run(experiment_id)

    detect_drift_task = SparkSubmitOperator(
        task_id="detect_drift",
        application="/opt/spark-apps/drift_detection.py",
        application_args=["--run-id", run_id],
        conn_id=SPARK_CONN_ID,
    )

    # Branching untuk retraining
    branching_task = BranchPythonOperator(
        task_id="check_drift",
        python_callable=check_drift_from_mlflow,
        op_args=["DRIFT_DETECTION"],
    )

    # Task jika retraining dilewati
    skip_retraining_task = EmptyOperator(task_id="skip_retraining")
    retrain_model_task = EmptyOperator(task_id="retrain_model")

    # Task selesai
    end_task = EmptyOperator(task_id="end")
    start_task = EmptyOperator(task_id="start")

    # Urutan dependensi
    start_task >> detect_drift_task >> branching_task
    branching_task >> retrain_model_task >> end_task
    branching_task >> skip_retraining_task >> end_task


detect_drift()
