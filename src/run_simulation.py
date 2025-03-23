# RunSimulation.py
from model_deployment_simulator import ModelDeploymentSimulator
from drift_detection_and_tracking import DriftDetectionAndMonitoring
import numpy as np
import joblib

# Load your processed data
X_train = np.load('data/processed/X_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_train = np.load('data/processed/y_train.npy')
y_val = np.load('data/processed/y_val.npy')

# Load the trained model
model = joblib.load('models/fashion_mnist_best_model.pkl')
pca = joblib.load('models/pca_model.pkl')
print("PCA is ",type(pca))
# Initialize the ModelDeploymentSimulator
deployment_simulator = ModelDeploymentSimulator(model, pca,X_val, y_val)

# Simulate data generation and drift
for _ in range(100):  # Simulating 10 rounds
    current_data = deployment_simulator.generate_new_data()  # Simulate new data
    current_accuracy = model.score(X_val, y_val)  # Get current accuracy
    deployment_simulator.monitor_performance(current_data, current_accuracy)  # Log performance

# Initialize DriftDetectionAndMonitoring
drift_monitor = DriftDetectionAndMonitoring(model, X_train, X_val, y_train, y_val)

# Simulate monitoring performance over time
for _ in range(100):  # Simulating 10 rounds of predictions
    current_data = deployment_simulator.generate_new_data()  # Simulate new data
    drift_monitor.track_performance_over_time(current_data,pca)  # Track performance and detect drift
