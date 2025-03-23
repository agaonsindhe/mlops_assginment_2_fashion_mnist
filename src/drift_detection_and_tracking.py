import mlflow
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score


class DriftDetectionAndMonitoring:
    def __init__(self, model, X_train, X_val, y_train, y_val):
        """
        Initialize with model and training/validation data.
        """
        self.model = model
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.previous_accuracy = None
        self.drift_threshold = 0.05

    def log_performance_metrics(self, accuracy, X_val, y_val, drift_detected):
        """ Log model performance metrics to MLflow dynamically """
        y_pred = self.model.predict(X_val)  # Get predictions on the validation set

        # Calculate metrics dynamically
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=1)

        # Log to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("drift detected", drift_detected)

    def detect_drift(self, current_accuracy):
        """ Compare current accuracy with previous and check for drift """
        if self.previous_accuracy is not None:
            accuracy_change = abs(current_accuracy - self.previous_accuracy)
            if accuracy_change > self.drift_threshold:
                print(f"Model drift detected: Accuracy changed from {self.previous_accuracy} to {current_accuracy}")
                return True
            else:
                print(f"No drift detected. Accuracy: {current_accuracy}")
        self.previous_accuracy = current_accuracy
        return False

    def trigger_retraining(self):
        """ Simulate retraining the model due to detected drift """
        print("Retraining triggered due to model drift.")
        self.model.fit(self.X_train, self.y_train)  # Retrain with new data
        joblib.dump(self.model, 'retrained_model.pkl')  # Save retrained model

        # Log retrained model in MLflow
        with mlflow.start_run():
            mlflow.log_metric("accuracy", self.model.score(self.X_val, self.y_val))  # Log new accuracy
            mlflow.sklearn.log_model(self.model, "retrained_model")
            mlflow.log_param("model_type", "MLPClassifier")  # Example parameter logging

    def track_performance_over_time(self, current_data, pca):
        """ Simulate tracking performance and detect drift """
        # Apply PCA transformation to the new data
        data_pca = pca.transform(current_data.reshape(1, -1))
        prediction = self.model.predict(data_pca)  # Get model prediction using transformed data
        current_accuracy = self.evaluate_model(data_pca)  # Or keep using raw data for evaluation if appropriate

        # Log performance and detect drift
        self.log_performance_metrics(current_accuracy,self.X_val, self.y_val,self.detect_drift(current_accuracy))
        if self.detect_drift(current_accuracy):
            print("Model drift detected. Retraining required.")
            # We can call trigger_retraining to simulate real time retraining whenever drift is detected.
        return current_accuracy

    def evaluate_model(self, data):
        """ Simulate model evaluation and calculate accuracy """
        return np.random.rand()  # Replace with actual evaluation logic
