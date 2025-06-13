import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.datasets import mnist
from mpl_toolkits.mplot3d import Axes3D

# 1. Load MNIST dataset
(x_train, y_train), (_, _) = mnist.load_data()

# 2. Preprocess: flatten images
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0

# 3. Subsample for speed (e.g., first 3000 samples)
n_samples = 3000
x_sample = x_train[:n_samples]
y_sample = y_train[:n_samples]

# 4. Run t-SNE in 3D
tsne = TSNE(n_components=3, perplexity=30, random_state=42, n_iter=1000, verbose=1)
x_tsne = tsne.fit_transform(x_sample)

# 5. Plot 3D scatter plot with classes colored
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    x_tsne[:, 0], x_tsne[:, 1], x_tsne[:, 2],
    c=y_sample, cmap=plt.cm.get_cmap("tab10", 10), s=20, alpha=0.8
)

ax.set_title('MNIST t-SNE 3D Visualization', fontsize=15)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_zlabel("t-SNE 3")

# Add legend
legend_labels = [str(i) for i in range(10)]
legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                             label=label, markerfacecolor=plt.cm.tab10(i/10), markersize=8)
                  for i, label in enumerate(legend_labels)]
ax.legend(handles=legend_handles, title="Digits")

plt.show()
