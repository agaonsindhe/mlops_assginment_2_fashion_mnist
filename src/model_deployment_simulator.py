import mlflow
import mlflow.sklearn
import numpy as np
import joblib
import logging
from sklearn.metrics import precision_score, recall_score, f1_score

# Set up logger for model performance
logging.basicConfig(
    filename='logs/model_performance.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set up MLflow experiment
mlflow.set_experiment("fashion_mnist_model_tracking")


class ModelDeploymentSimulator:
    def __init__(self, model, pca, X_val, y_val):
        self.model = model  # Loaded model
        self.pca = pca  # Loaded PCA model
        self.X_val = X_val  # Validation data
        self.y_val = y_val  # True labels
        self.previous_accuracy = None

    def generate_new_data(self):
        """ Generate new data to simulate incoming requests """
        return np.random.rand(784)  # Example for Fashion MNIST

    def simulate_drift(self, data):
        """ Simulate drift by adding noise to data """
        return data + np.random.normal(0, 0.1, data.shape)  # Add noise to simulate drift

    def serve_prediction(self, data):
        """ Simulate prediction by the deployed model """
        # Apply PCA transformation to incoming data
        data_pca = self.pca.transform([data])  # Transform new data to match the training data shape
        prediction = self.model.predict(data_pca)
        return prediction

    def monitor_performance(self, current_data, current_accuracy):
        """ Log performance over time using MLflow """
        # Apply PCA transformation to incoming data
        data_pca = self.pca.transform([current_data])  # Transform new data for prediction

        # Get prediction from model
        prediction = self.serve_prediction(current_data)

        # Log prediction
        logging.info(f"Prediction: {prediction[0]}")  # Log the prediction (first item in prediction array)

        # Log performance metrics using MLflow
        mlflow.log_metric("accuracy", current_accuracy)
        mlflow.log_metric("drift_detected", int(self.detect_drift(current_accuracy)))
        self.previous_accuracy = current_accuracy
        return current_accuracy

    def detect_drift(self, current_accuracy):
        """ Check for drift in the model's performance """
        if self.previous_accuracy and abs(self.previous_accuracy - current_accuracy) > 0.05:  # Example threshold
            return True
        return False
