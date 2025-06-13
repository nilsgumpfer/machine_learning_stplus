import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import umap
from mpl_toolkits.mplot3d import Axes3D

# 1. Load MNIST dataset
(x_train, y_train), (_, _) = mnist.load_data()

# 2. Preprocess: flatten images
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0

# 3. Subsample for speed (e.g., first 3000 samples)
n_samples = 3000
x_sample = x_train[:n_samples]
y_sample = y_train[:n_samples]

# 4. Run UMAP in 3D
reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1, random_state=42)
x_umap = reducer.fit_transform(x_sample)

# 5. Plot 3D scatter plot with classes colored
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    x_umap[:, 0], x_umap[:, 1], x_umap[:, 2],
    c=y_sample, cmap=plt.cm.get_cmap("tab10", 10), s=20, alpha=0.8
)

ax.set_title('MNIST UMAP 3D Visualization', fontsize=15)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_zlabel("UMAP 3")

# Add legend
legend_labels = [str(i) for i in range(10)]
legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                             label=label, markerfacecolor=plt.cm.tab10(i/10), markersize=8)
                  for i, label in enumerate(legend_labels)]
ax.legend(handles=legend_handles, title="Digits")

plt.show()
