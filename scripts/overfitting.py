import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# 1. Create noisy data
np.random.seed(0)
x = np.linspace(0, 2 * np.pi, 30)
# x = np.linspace(0, 2 * np.pi, 150)
y_true = np.sin(x)
noise = np.random.normal(0, 0.2, size=x.shape)
y = y_true + noise

# 2. High-resolution x for plotting models
x_fit = np.linspace(0, 2 * np.pi, 500)


def fit_and_plot(degree, ax_model):
    """Fits a polynomial of the given degree and plots it."""
    # Fit polynomial
    coefs = np.polyfit(x, y, degree)
    poly = np.poly1d(coefs)
    y_fit = poly(x_fit)

    # Plot
    ax_model.clear()
    ax_model.plot(x, y, 'bo', label="Noisy Data")
    ax_model.plot(x_fit, np.sin(x_fit), 'g--', label="True sin(x)")
    ax_model.plot(x_fit, y_fit, 'r-', label=f"Poly deg {degree}")
    ax_model.set_ylim(-2, 2)
    ax_model.set_title("Polynomial Fit to Noisy sin(x)")
    ax_model.legend()
    ax_model.grid(True)


# 3. Set up plot
fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(bottom=0.25)

# Initial plot
initial_degree = 3
fit_and_plot(initial_degree, ax)

# 4. Add slider
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'Degree', valmin=1, valmax=50, valinit=initial_degree, valstep=1)


def update(val):
    degree = int(slider.val)
    fit_and_plot(degree, ax)
    fig.canvas.draw_idle()


slider.on_changed(update)

plt.show()
