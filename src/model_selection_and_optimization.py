import numpy as np
import pandas as pd
from tpot import TPOTClassifier
import optuna
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


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

    def automl_model_selection(self):
        """
        Use AutoML (TPOT) to select the best model.
        """
        print("Starting AutoML model selection using TPOT...")

        # Initialize TPOTClassifier with generations and population size for optimization
        tpot = TPOTClassifier(verbosity=2, generations=5, population_size=20, random_state=42)

        # Fit the model using training data
        tpot.fit(self.X_train, self.y_train)

        # Output the best pipeline and score
        print(f"Best pipeline found by TPOT: {tpot.fitted_pipeline_}")
        print(f"Best accuracy on validation set: {tpot.score(self.X_val, self.y_val)}")

        # Save the best model
        self.best_model = tpot.fitted_pipeline_

        # Export the best model
        tpot.export('best_model_pipeline.py')

        return tpot

    def hyperparameter_optimization(self, model=None):
        """
        Use Optuna for hyperparameter optimization.
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
            max_iter = trial.suggest_int('max_iter', 200, 1000, step=100)

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
        Evaluate the final model on the validation set.
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
        import joblib
        joblib.dump(model, 'fashion_mnist_best_model.pkl')
        print("✅ Best Model saved as 'fashion_mnist_best_model.pkl'")


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


if __name__ == "__main__":
    main()
