import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load Iris dataset (Setosa vs Others)
iris = load_iris()
X = iris.data[:, :2]  # Sepal length and width
y = (iris.target == 0).astype(int)  # Setosa = 1, others = 0

# Initial weights and bias
w = np.array([1.0, -1.0])
b = 0.0

# Create figure
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)
ax.set_title("Linear Classifier")
ax.set_xlabel("sepal length")
ax.set_ylabel("sepal width")

#
# Grid for background
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300)
)

# Plot data
ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Setosa')
ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='Others')

# Decision boundary line
boundary, = ax.plot([], [], 'k-', linewidth=1)

# Accuracy text
accuracy_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=10, verticalalignment='top')

# Track the background fill
bg = None

# Update function
def update(w1, w2, b_val):
    global bg
    w = np.array([w1, w2])
    b = b_val

    # Remove old background (cleanly and officially)
    if bg is not None:
        bg.remove()

    # Compute background classification
    Z = (w[0] * xx + w[1] * yy + b) < 0
    bg = ax.contourf(xx, yy, Z, levels=[-1, 0, 1], colors=["blue", "red"], alpha=0.25)

    # Update decision boundary
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    if w[1] != 0:
        y_vals = -(w[0] * x_vals + b) / w[1]
        boundary.set_data(x_vals, y_vals)
    else:
        boundary.set_data([], [])

    # Prediction
    preds = (X @ w + b) < 0  # Predict 1 (Setosa) if value < 0

    # Confusion matrix components
    TP = np.sum((preds == 1) & (y == 1))  # True Setosa
    TN = np.sum((preds == 0) & (y == 0))  # True Other
    FP = np.sum((preds == 1) & (y == 0))  # Predicted Setosa, was Other
    FN = np.sum((preds == 0) & (y == 1))  # Predicted Other, was Setosa

    # Sensitivity and Specificity
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    # Display
    accuracy_text.set_text(f"Sens: {sensitivity:.2f}  Spec: {specificity:.2f}")

# Initial draw
update(w[0], w[1], b)

# Sliders
ax_w1 = plt.axes([0.2, 0.25, 0.6, 0.03])
ax_w2 = plt.axes([0.2, 0.2, 0.6, 0.03])
ax_b = plt.axes([0.2, 0.15, 0.6, 0.03])

slider_w1 = Slider(ax_w1, 'w1', -10.0, 10.0, valinit=w[0])
slider_w2 = Slider(ax_w2, 'w2', -10.0, 10.0, valinit=w[1])
slider_b = Slider(ax_b, 'b', -10.0, 10.0, valinit=b)

slider_w1.on_changed(lambda val: update(slider_w1.val, slider_w2.val, slider_b.val))
slider_w2.on_changed(lambda val: update(slider_w1.val, slider_w2.val, slider_b.val))
slider_b.on_changed(lambda val: update(slider_w1.val, slider_w2.val, slider_b.val))

ax.legend(loc='upper right')
plt.show()
