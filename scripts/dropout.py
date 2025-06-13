import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# Network sizes
n_input = 3
n_hidden = 10
n_output = 2

layer_x = [0, 1, 2]
layer_sizes = [n_input, n_hidden, n_output]

# Generate node positions
node_positions = []
for i, size in enumerate(layer_sizes):
    y = np.linspace(0.2, 0.8, size)
    x = np.full_like(y, layer_x[i], dtype=float)
    node_positions.append(np.stack((x, y), axis=1))

fig, ax = plt.subplots(figsize=(7, 5))
plt.subplots_adjust(bottom=0.25)
ax.axis('off')

# Slider for dropout rate
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'Dropout rate', 0.0, 1.0, valinit=0.5, valstep=0.05)

# Draw the full network with dropout mask
def draw_network(drop_mask=None):
    ax.clear()
    ax.axis('off')

    for i in range(len(layer_sizes)):
        positions = node_positions[i]

        if i == 1 and drop_mask is not None:
            mask = drop_mask
        else:
            mask = np.ones(len(positions), dtype=bool)

        color = 'gray'
        alpha = 1.0 if i != 1 else mask.astype(float)

        ax.scatter(positions[:, 0], positions[:, 1], s=300, color=color, alpha=alpha, zorder=3)

        # Draw connections
        if i < len(layer_sizes) - 1:
            from_layer = node_positions[i]
            to_layer = node_positions[i + 1]
            for j, from_pos in enumerate(from_layer):
                for k, to_pos in enumerate(to_layer):
                    if i == 0:
                        # Always connect input → hidden
                        ax.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]],
                                color='black', alpha=0.3, zorder=1)
                    elif i == 1 and drop_mask is not None and drop_mask[j]:
                        # Only connect active hidden → output
                        ax.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]],
                                color='black', alpha=0.3, zorder=1)

# Animate dropout
def animate(frame):
    dropout_rate = slider.val
    k = int(round((1 - dropout_rate) * n_hidden))
    drop_mask = np.zeros(n_hidden, dtype=bool)
    drop_mask[np.random.choice(n_hidden, k, replace=False)] = True
    draw_network(drop_mask)

# Connect slider to update
slider.on_changed(lambda val: draw_network(np.ones(n_hidden, dtype=bool)))

# Initial draw and animation
draw_network(np.ones(n_hidden, dtype=bool))
ani = FuncAnimation(fig, animate, interval=1000)

plt.show()
