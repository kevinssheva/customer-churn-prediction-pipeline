import random
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import subprocess

def run_spark_script(script_path):
    """Runs a Spark script using spark-submit."""
    try:
        subprocess.check_call(['spark-submit', script_path])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to run Spark script: {script_path}. Error: {e}")

def run_mlflow_script(script_path):
    """Runs an MLflow script."""
    try:
        subprocess.check_call(['python', script_path])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to run MLflow script: {script_path}. Error: {e}")

# Function to simulate data drift
def simulate_drift():
    """
    Simulates data drift by modifying a dataset's features.
    This function creates a synthetic dataset and alters it to mimic drift.
    """
    # Original distribution
    original_data = pd.DataFrame({
        'feature_1': np.random.normal(50, 5, 1000),
        'feature_2': np.random.uniform(20, 80, 1000),
        'label': np.random.choice([0, 1], size=1000, p=[0.7, 0.3]),
    })
    
    # Simulate drift by changing feature distributions
    drifted_data = pd.DataFrame({
        'feature_1': np.random.normal(60, 10, 1000),  # Shift mean and increase std dev
        'feature_2': np.random.uniform(30, 90, 1000),  # Shift range
        'label': np.random.choice([0, 1], size=1000, p=[0.5, 0.5]),  # Change class balance
    })
    
    # Save the datasets
    original_data.to_csv('/tmp/original_data.csv', index=False)
    drifted_data.to_csv('/tmp/drifted_data.csv', index=False)
    print("Data drift simulation completed.")

# Function to monitor data drift
def monitor_drift():
    """
    Detects data drift using Population Stability Index (PSI).
    Compares the original dataset to the drifted dataset.
    """
    # Load datasets
    original_data = pd.read_csv('/tmp/original_data.csv')
    drifted_data = pd.read_csv('/tmp/drifted_data.csv')
    
    def calculate_psi(base, current, bins=10):
        """
        Calculate Population Stability Index (PSI) between two distributions.
        """
        base_hist, bin_edges = np.histogram(base, bins=bins)
        current_hist, _ = np.histogram(current, bins=bin_edges)
        
        # Normalize the distributions
        base_dist = base_hist / len(base)
        current_dist = current_hist / len(current)
        
        # Avoid divide by zero errors
        base_dist = np.where(base_dist == 0, 1e-8, base_dist)
        current_dist = np.where(current_dist == 0, 1e-8, current_dist)
        
        # Calculate PSI
        psi = np.sum((base_dist - current_dist) * np.log(base_dist / current_dist))
        return psi
    
    # Calculate PSI for each feature
    psi_results = {}
    for column in ['feature_1', 'feature_2']:
        psi = calculate_psi(original_data[column], drifted_data[column])
        psi_results[column] = psi
    
    # Print PSI results
    for feature, psi in psi_results.items():
        print(f"PSI for {feature}: {psi:.4f}")
    
    # Flagging significant drift
    if any(psi > 0.2 for psi in psi_results.values()):
        print("Significant data drift detected! Consider retraining the model.")
    else:
        print("No significant data drift detected.")
