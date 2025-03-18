import numpy as np
import os
from tensorflow.keras.datasets import fashion_mnist

# Create raw data directory if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Save dataset as NumPy arrays
np.save("data/raw/train_images.npy", train_images)
np.save("data/raw/train_labels.npy", train_labels)
np.save("data/raw/test_images.npy", test_images)
np.save("data/raw/test_labels.npy", test_labels)

print("✅ Fashion MNIST dataset downloaded and saved successfully!")
