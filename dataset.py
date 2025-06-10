import os
import numpy as np
from PIL import Image
import string
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# data directory (for your use download the dataset from Kaggle)
data_dir = './captchas'

# Define the character set (lowercase letters + digits)
characters = string.ascii_lowercase + string.digits
char_to_idx = {char: idx for idx, char in enumerate(characters)}

# Encode labels as indices
def encode_label(label):
    return [char_to_idx[char] for char in label]

# Get all image files in the directory
all_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# images and labels
images = []
labels = []

for img_file in all_files:
    label = os.path.splitext(img_file.split('_')[1])[0].lower()
    img_path = os.path.join(data_dir, img_file)
    image = Image.open(img_path).convert('L')  # Convert to grayscale
    image = image.resize((250, 50))  # Resize to consistent shape
    images.append(np.array(image))
    labels.append(encode_label(label))  # Convert label to list of indices

# to NumPy arrays
images = np.array(images) / 255.0
labels = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32).reshape(-1, 1, 50, 250)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]  # label is a sequence
