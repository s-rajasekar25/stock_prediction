import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
data = iris.data
target = iris.target

# Standardize data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Visualize reduced data
plt.figure(figsize=(8,6))
plt.scatter(data_pca[:,0], data_pca[:,1], c=target, cmap='viridis', edgecolors='k')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Iris Dataset Reduced to 2D")
plt.colorbar(label="Target Classes")
plt.show()