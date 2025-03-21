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

# Sample 10% of the data for quick processing
df_sample = df.sample(frac=0.1, random_state=42)  # 10% of the data

# Verify the size
print(f"Sampled data size: {df_sample.shape}")
# Generate EDA report using ydata-profiling
profile = ProfileReport(df_sample, title="Fashion MNIST EDA Report", explorative=True, minimal=True)
profile.to_file("reports/eda_report.html")

print("✅ EDA report generated: reports/eda_report.html")
