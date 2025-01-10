import mlflow
from subprocess import call

# Check drift and retrain
def retrain_model():
    if check_drift():  # Implement a drift check using PSI
        call(['spark-submit', 'spark/train_model.py'])
