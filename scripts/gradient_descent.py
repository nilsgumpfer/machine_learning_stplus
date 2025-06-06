import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Parameters for elliptical quarry shape
a = 20.0   # x1 semi-axis (wide direction)
b = 5.0   # x2 semi-axis (narrow direction)
k = 3.0   # Steepness of rim
R = 1.0   # Radius at which the walls begin

def elliptical_r(x):
    return np.sqrt((x[0]/a)**2 + (x[1]/b)**2)

A = 100.0  # scale factor for function height

def f(x):
    r = elliptical_r(x)
    return A / (1 + np.exp(-k * (r - R)))

def grad_f(x):
    x1, x2 = x
    denom = np.sqrt((x1/a)**2 + (x2/b)**2)
    if denom == 0:
        return np.array([0.0, 0.0])  # avoid divide by zero

    exp_term = np.exp(-k * (denom - R))
    coeff = A * (k * exp_term) / ((1 + exp_term)**2 * denom)

    df_dx1 = coeff * (x1 / a**2)
    df_dx2 = coeff * (x2 / b**2)
    return np.array([df_dx1, df_dx2])

def compute_path(alpha, num_steps):
    x = np.array([10, -12])  # Start at canyon rim
    path = [x.copy()]
    for _ in range(num_steps - 1):
        grad = grad_f(x)
        x = x - alpha * grad
        path.append(x.copy())
    return np.array(path)

# Surface data
X = np.linspace(-10, 30, 50)
Y = np.linspace(-13, 13, 50)
X_grid, Y_grid = np.meshgrid(X, Y)
R_grid = np.sqrt((X_grid/a)**2 + (Y_grid/b)**2)
Z = A / (1 + np.exp(-k * (R_grid - R)))

# Initial values
init_alpha = 0.04
init_steps = 280
path = compute_path(init_alpha, 500)
fx_path = np.array([f(p) for p in path])

# Plot setup
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', alpha=0.6)

line, = ax.plot([], [], [], 'r-', lw=2)
start_point = ax.scatter([], [], [], color='blue', s=50)
current_point = ax.scatter([], [], [], color='red', s=50)

ax.set_xlim(-10, 30)
ax.set_ylim(-13, 13)
ax.set_zlim(0, A * 1.1)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x)$')
ax.set_title("Gradient Descent into an Oval Canyon")

# Slider axes
slider_ax_step = plt.axes([0.25, 0.05, 0.65, 0.03])
slider_ax_azim = plt.axes([0.25, 0.01, 0.30, 0.03])
slider_ax_elev = plt.axes([0.65, 0.01, 0.30, 0.03])
slider_ax_alpha = plt.axes([0.25, 0.09, 0.65, 0.03])

slider_step = Slider(slider_ax_step, 'Step', 2, 500, valinit=init_steps, valstep=1)
slider_azim = Slider(slider_ax_azim, 'Azimuth', 0, 360, valinit=164)
slider_elev = Slider(slider_ax_elev, 'Elevation', 0, 90, valinit=31)
slider_alpha = Slider(slider_ax_alpha, 'Learning Rate', 0.01, 0.8, valinit=init_alpha, valstep=0.01)

def update(val):
    step = int(slider_step.val)
    azim = slider_azim.val
    elev = slider_elev.val
    alpha = slider_alpha.val

    global path, fx_path
    path = compute_path(alpha, 500)
    fx_path = np.array([f(p) for p in path])

    line.set_data(path[:step, 0], path[:step, 1])
    line.set_3d_properties(fx_path[:step])

    start_point._offsets3d = ([path[0, 0]], [path[0, 1]], [fx_path[0]])
    current_point._offsets3d = ([path[step-1, 0]], [path[step-1, 1]], [fx_path[step-1]])

    ax.view_init(elev=elev, azim=azim)
    fig.canvas.draw_idle()

# Connect sliders
slider_step.on_changed(update)
slider_azim.on_changed(update)
slider_elev.on_changed(update)
slider_alpha.on_changed(update)

# Initial draw
update(None)
plt.show()
