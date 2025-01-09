import mlflow
from mlflow.tracking import MlflowClient

# Log additional metadata
mlflow.set_experiment('telco_churn')
with mlflow.start_run():
    mlflow.log_param('data_version', '1.0')
    mlflow.log_metric('roc_auc', 0.85)
