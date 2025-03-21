import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class FeatureEngineeringPipeline:
    def __init__(self, train_images, train_labels, processed_dir="data/processed", plot_dir="data/feature_importance"):
        """
        Initializes with raw images and labels, as well as directories to save processed data and plots.
        """
        self.train_images = train_images
        self.train_labels = train_labels
        self.processed_dir = processed_dir  # Directory to store processed data
        self.plot_dir = plot_dir  # Directory to store feature importance plots
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.pca = None

        # Create the plot directory if it doesn't exist
        os.makedirs(self.plot_dir, exist_ok=True)

    def preprocess_data(self):
        """ Normalize the data and flatten the images """
        # Normalize the images to [0, 1]
        self.train_images = self.train_images / 255.0
        # Flatten the images (28x28 -> 784 features per image)
        X = self.train_images.reshape(self.train_images.shape[0], -1)
        return X

    def split_data(self, test_size=0.2):
        """ Split the data into training and validation sets """
        X = self.preprocess_data()
        # Split the data into training and validation sets
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, self.train_labels, test_size=test_size,
                                                                              random_state=42)
        print(f"X_train shape: {self.X_train.shape}")
        print(f"X_val shape: {self.X_val.shape}")

    def remove_highly_correlated_features(self, threshold=0.9):
        """ Remove highly correlated features from the data """
        corr_matrix = pd.DataFrame(self.X_train).corr().abs()  # Calculate the correlation matrix (absolute values)
        upper_tri = np.triu(corr_matrix, k=1)  # Upper triangular matrix (to avoid redundancy)
        to_drop = [column for column in corr_matrix.columns if any(corr_matrix[column] > threshold)]

        print(f"Removing {len(to_drop)} highly correlated features")

        # Ensure we don't remove all features, check before dropping
        if len(to_drop) < self.X_train.shape[1]:
            self.X_train = pd.DataFrame(self.X_train).drop(columns=to_drop).values
            self.X_val = pd.DataFrame(self.X_val).drop(columns=to_drop).values
        else:
            print("Warning: Removing all features due to high correlation. Adjust threshold.")

    def apply_pca(self, n_components=0.95):
        """ Apply PCA for dimensionality reduction """
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=n_components)
        self.X_train = self.pca.fit_transform(self.X_train)
        self.X_val = self.pca.transform(self.X_val)
        print(f"Number of features after PCA: {self.X_train.shape[1]}")

    def save_processed_data(self):
        """ Save processed data to the 'processed' folder """
        # Ensure the processed directory exists
        os.makedirs(self.processed_dir, exist_ok=True)

        # Save processed data (X_train, X_val, y_train, y_val)
        np.save(os.path.join(self.processed_dir, "X_train.npy"), self.X_train)
        np.save(os.path.join(self.processed_dir, "X_val.npy"), self.X_val)
        np.save(os.path.join(self.processed_dir, "y_train.npy"), self.y_train)
        np.save(os.path.join(self.processed_dir, "y_val.npy"), self.y_val)

        print(f"✅ Processed data saved in '{self.processed_dir}'.")

    def visualize_feature_importance(self):
        """ Visualize feature importance based on feature engineering decisions (Normalization, PCA, etc.) """
        # Visualizing the effect of normalization
        plt.hist(self.X_train.flatten(), bins=50, alpha=0.5, label="Normalized Features")
        plt.legend()
        plt.title("Effect of Normalization on Feature Distribution")
        plt.savefig(os.path.join(self.plot_dir, "effect_of_normalization.png"))  # Save plot as image
        plt.close()

        # Visualizing the explained variance after PCA
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance')
        plt.title('Explained Variance vs. Number of PCA Components')
        plt.savefig(os.path.join(self.plot_dir, "explained_variance_pca.png"))  # Save plot as image
        plt.close()


def main():
    # Load the dataset
    train_images = np.load("data/raw/train_images.npy")
    train_labels = np.load("data/raw/train_labels.npy")

    # Initialize the pipeline
    fe_pipeline = FeatureEngineeringPipeline(train_images, train_labels)

    # Step 1: Split the data into train and validation sets
    fe_pipeline.split_data(test_size=0.2)

    # Step 2: Remove highly correlated features
    fe_pipeline.remove_highly_correlated_features(threshold=0.9)

    # Step 3: Apply PCA for dimensionality reduction
    fe_pipeline.apply_pca(n_components=0.95)

    # Step 4: Save processed data in the processed/ folder
    fe_pipeline.save_processed_data()

    # Step 5: Visualize the feature importance (based on feature engineering decisions)
    fe_pipeline.visualize_feature_importance()


# Ensure the main function runs when the script is executed
if __name__ == "__main__":
    main()
