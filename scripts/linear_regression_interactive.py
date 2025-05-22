import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import gridspec

# Generate synthetic data
np.random.seed(0)
x = np.linspace(0, 2, 20)
y = 0.8 * x + 0.5 + 0.2 * np.random.randn(20)

# Gradient descent setup
theta0, theta1 = 0.0, 0.0
learning_rate = 0.1
steps = 50
params = [(theta0, theta1)]

# Compute loss surface
theta0_range = np.linspace(0, 2, 100)
theta1_range = np.linspace(-1, 1, 100)
Theta0, Theta1 = np.meshgrid(theta0_range, theta1_range)
Loss = np.zeros_like(Theta0)
for i in range(Theta0.shape[0]):
    for j in range(Theta0.shape[1]):
        y_pred = Theta0[i, j] + Theta1[i, j] * x
        Loss[i, j] = np.mean((y - y_pred) ** 2)

# Simulate training
for _ in range(steps):
    y_pred = theta0 + theta1 * x
    dtheta0 = -2 * np.mean(y - y_pred)
    dtheta1 = -2 * np.mean((y - y_pred) * x)
    theta0 -= learning_rate * dtheta0
    theta1 -= learning_rate * dtheta1
    params.append((theta0, theta1))

# Set up the figure and axes
fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
plt.subplots_adjust(bottom=0.25)

# Plot loss surface with viridis colormap
cs = ax1.contourf(Theta0, Theta1, Loss, levels=50, cmap='viridis')
fig.colorbar(cs, ax=ax1, label='Loss')

path, = ax1.plot([], [], 'o-', color='white')
ax1.set_xlabel("Intercept, $\\theta_0$")
ax1.set_ylabel("Slope, $\\theta_1$")
ax1.set_title("a) Loss, $L[\\theta]$")

# Plot regression lines
scatter = ax2.scatter(x, y, color='darkorange')
lines = [ax2.plot(x, p[0] + p[1]*x, color='skyblue', alpha=0.2)[0] for p in params]
highlight, = ax2.plot([], [], color='dodgerblue', lw=3)
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 2)
ax2.set_xlabel("Input, $x$")
ax2.set_ylabel("Output, $y$")
ax2.set_title("b) Regression fit")

# Add slider
ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
slider = Slider(ax_slider, 'Step', 0, steps, valinit=0, valstep=1)

def update(step):
    coords = np.array(params[:step+1])
    path.set_data(coords[:, 0], coords[:, 1])
    highlight.set_data(x, params[step][0] + params[step][1]*x)
    for i, line in enumerate(lines):
        line.set_alpha(1.0 if i == step else 0.2)
    fig.canvas.draw_idle()

slider.on_changed(update)
update(0)

plt.show()
