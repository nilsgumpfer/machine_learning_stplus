import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from tensorflow import keras
from tensorflow.keras import layers

# Define the true polynomial function
def true_function(x):
    return 0.5 * x**3 - x**2 + 2 * x + 1

# Create training data
np.random.seed(42)
x_train = np.linspace(-3, 3, 300)
y_train = true_function(x_train)
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

# Prepare for model
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

# Function to create and train the model
def create_and_train_model(units1, units2, activation):
    model = keras.Sequential()
    model.add(layers.Dense(units1, activation=activation, input_shape=(1,)))
    if units2 > 0:
        model.add(layers.Dense(units2, activation=activation))
    model.add(layers.Dense(1))  # output
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train, epochs=300, batch_size=32, verbose=0)
    return model

# Function to draw the network architecture and weights
def draw_network(model, ax):
    ax.clear()
    ax.axis("off")

    layer_sizes = []
    weights = []
    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            W, b = layer.get_weights()
            weights.append(W)
            layer_sizes.append(W.shape[0])
    layer_sizes.append(1)  # output layer

    # Positions
    v_spacing = 1.0
    h_spacing = 2.0
    x_coords = np.arange(len(layer_sizes)) * h_spacing
    node_positions = []

    # Compute node y-positions
    for i, n_nodes in enumerate(layer_sizes):
        y = np.linspace(-v_spacing * (n_nodes - 1) / 2, v_spacing * (n_nodes - 1) / 2, n_nodes)
        node_positions.append((x_coords[i] * np.ones_like(y), y))

    # Draw connections
    for i, W in enumerate(weights):
        x0, y0 = node_positions[i]
        x1, y1 = node_positions[i + 1]
        for j in range(W.shape[0]):
            for k in range(W.shape[1]):
                w = W[j, k]
                ax.plot([x0[j], x1[k]], [y0[j], y1[k]], color='blue' if w < 0 else 'red',
                        linewidth=np.clip(abs(w) * 3, 0.5, 5), alpha=0.8)

    # Draw nodes
    for (x, y) in node_positions:
        ax.scatter(x, y, s=300, c='white', edgecolors='black', zorder=3)

    ax.set_title("Learned Network Weights", fontsize=12)

# Initial config
initial_units1 = 3
initial_units2 = 0
initial_activation = 'relu'
model = create_and_train_model(initial_units1, initial_units2, initial_activation)
y_pred = model.predict(x_train).flatten()

# Layout
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
ax_plot = fig.add_subplot(gs[0])
ax_net = fig.add_subplot(gs[1])
plt.subplots_adjust(left=0.25, bottom=0.35)

line_true, = ax_plot.plot(x_train, y_train, label="True Polynomial", linestyle="--", color='green')
line_pred, = ax_plot.plot(x_train, y_pred, label="NN Approximation", linewidth=2, color='orange')
ax_plot.set_title("Polynomial Approximation with Neural Network")
ax_plot.set_xlabel("x")
ax_plot.set_ylabel("f(x)")
ax_plot.legend()
ax_plot.grid(True)

draw_network(model, ax_net)

# Sliders
ax_units1 = plt.axes([0.25, 0.25, 0.65, 0.03])
slider_units1 = Slider(ax_units1, 'Hidden Layer 1 Units', 1, 50, valinit=initial_units1, valstep=1)

ax_units2 = plt.axes([0.25, 0.2, 0.65, 0.03])
slider_units2 = Slider(ax_units2, 'Hidden Layer 2 Units', 0, 50, valinit=initial_units2, valstep=1)

# Activation selector
ax_activation = plt.axes([0.025, 0.5, 0.15, 0.25])
radio_activation = RadioButtons(ax_activation, ('relu', 'tanh', 'softplus', 'elu'), active=0)

# Update function
def update(val):
    units1 = int(slider_units1.val)
    units2 = int(slider_units2.val)
    activation = radio_activation.value_selected
    global model
    model = create_and_train_model(units1, units2, activation)
    y_pred = model.predict(x_train).flatten()
    line_pred.set_ydata(y_pred)
    draw_network(model, ax_net)
    fig.canvas.draw_idle()

slider_units1.on_changed(update)
slider_units2.on_changed(update)
radio_activation.on_clicked(update)

plt.show()
