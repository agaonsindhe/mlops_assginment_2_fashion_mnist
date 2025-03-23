import numpy as np
import pandas as pd
from tpot import TPOTClassifier
import optuna
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

import logging
import warnings
warnings.filterwarnings("ignore", message="Warning: optional dependency `torch` is not available.")

# Set up logger
logging.basicConfig(
    filename='model_selection_and_optimization.log',  # Log file name
    level=logging.INFO,  # Log level (INFO will capture everything)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

class ModelSelectionAndOptimization:
    def __init__(self, X_train, X_val, y_train, y_val):
        """
        Initialize with training and validation data.
        """
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.best_model = None
        self.tpot_results = None
        self.optuna_results = None

    def automl_model_selection(self):
        """
        Use AutoML (TPOT) to select the best model and capture AutoML results.
        """
        print("Starting AutoML model selection using TPOT...")

        # Initialize TPOTClassifier with generations and population size for optimization
        tpot = TPOTClassifier(verbosity=2, generations=3, population_size=10, random_state=42, n_jobs=-1)

        # Fit the model using training data
        tpot.fit(self.X_train, self.y_train)

        # Capture and print the best model and its performance
        print(f"Best pipeline found by TPOT: {tpot.fitted_pipeline_}")
        print(f"Best accuracy on validation set: {tpot.score(self.X_val, self.y_val)}")

        # Save the best model
        self.best_model = tpot.fitted_pipeline_

        # Store AutoML results
        self.tpot_results = {
            "best_model_pipeline": tpot.fitted_pipeline_,
            "best_accuracy": tpot.score(self.X_val, self.y_val)
        }

        # Export the best model
        tpot.export('best_model_pipeline.py')

        return tpot

    def hyperparameter_optimization(self, model=None):
        """
        Use Optuna for hyperparameter optimization and capture the tuning logs.
        """
        if model is None:
            model = self.best_model

        print("Starting hyperparameter optimization using Optuna...")

        def objective(trial):
            """
            Define the objective function for Optuna optimization.
            """
            # Define hyperparameters for optimization
            hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (150,)])
            activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
            solver = trial.suggest_categorical('solver', ['adam', 'sgd'])
            learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
            max_iter = trial.suggest_int('max_iter', 50, 200, step=50)

            # Create and train the model
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                learning_rate=learning_rate,
                max_iter=max_iter,
                random_state=42
            )

            # Evaluate the model using cross-validation
            score = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            return score.mean()

        # Use Optuna to optimize hyperparameters
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        # Store hyperparameter tuning logs
        self.optuna_results = {
            "best_trial_params": study.best_trial.params,
            "best_trial_value": study.best_trial.value
        }

        print(f"Best trial params: {study.best_trial.params}")

        # Train the model with the best hyperparameters
        best_params = study.best_trial.params
        optimized_model = MLPClassifier(
            hidden_layer_sizes=best_params['hidden_layer_sizes'],
            activation=best_params['activation'],
            solver=best_params['solver'],
            learning_rate=best_params['learning_rate'],
            max_iter=best_params['max_iter'],
            random_state=42
        )
        optimized_model.fit(self.X_train, self.y_train)

        # Evaluate the optimized model
        optimized_accuracy = optimized_model.score(self.X_val, self.y_val)
        print(f"Optimized Model Accuracy on Validation Set: {optimized_accuracy}")

        return optimized_model

    def evaluate_model(self, model=None):
        """
        Evaluate the final model on the validation set and capture the results.
        """
        if model is None:
            model = self.best_model

        accuracy = model.score(self.X_val, self.y_val)
        print(f"Model Accuracy on Validation Set: {accuracy}")

        return accuracy

    def save_best_model(self, model=None):
        """ Save the optimized model to a file """
        if model is None:
            model = self.best_model

        # Save the model using joblib
        joblib.dump(model, 'fashion_mnist_best_model.pkl')
        print("✅ Best Model saved as 'fashion_mnist_best_model.pkl'")

    def print_deliverables(self):
        """ Log the deliverables for M3 """

        logging.info("\n--- AutoML Results ---")
        logging.info(f"Best Model Pipeline: {self.tpot_results['best_model_pipeline']}")
        logging.info(f"Best Accuracy on Validation Set: {self.tpot_results['best_accuracy']}")

        logging.info("\n--- Hyperparameter Tuning Logs ---")
        logging.info(f"Best Hyperparameters: {self.optuna_results['best_trial_params']}")
        logging.info(f"Best Trial Value: {self.optuna_results['best_trial_value']}")

        logging.info("\n--- Justification for Chosen Model ---")
        logging.info("The chosen model is based on AutoML results using TPOT, which selected the best pipeline. "
                     "The best hyperparameters were optimized using Optuna. "
                     "The performance of the final model was evaluated on the validation set, achieving the best accuracy.")


def main():
    # Load the processed data
    X_train = np.load('data/processed/X_train.npy')
    X_val = np.load('data/processed/X_val.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_val = np.load('data/processed/y_val.npy')

    # Initialize the model selection and optimization class
    model_selection = ModelSelectionAndOptimization(X_train, X_val, y_train, y_val)

    # Step 1: Use AutoML (TPOT) for model selection
    model_selection.automl_model_selection()

    # Step 2: Hyperparameter Optimization using Optuna
    optimized_model = model_selection.hyperparameter_optimization()

    # Step 3: Evaluate the final optimized model
    model_selection.evaluate_model(optimized_model)

    # Step 4: Save the best model
    model_selection.save_best_model(optimized_model)

    # Step 5: Print the deliverables
    model_selection.print_deliverables()


if __name__ == "__main__":
    main()
