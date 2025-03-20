import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import os

# Load dataset
train_images = np.load("data/raw/train_images.npy")
train_labels = np.load("data/raw/train_labels.npy")

# Flatten images into 1D arrays and add labels as a column
flattened_images = train_images.reshape(train_images.shape[0], -1)
df = pd.DataFrame(flattened_images)  # Flatten the image data
df['label'] = train_labels  # Add the labels column

# Generate EDA report using ydata-profiling
profile = ProfileReport(df, title="Fashion MNIST EDA Report", explorative=True)
profile.to_file("reports/eda_report.html")

print("✅ EDA report generated: reports/eda_report.html")
