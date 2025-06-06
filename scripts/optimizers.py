import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import cm


# Define the loss function and gradient
def loss_fn(x, y):
    return x ** 2 / 20.0 + y ** 2


def grad_fn(x, y):
    return np.array([x / 10.0, 2.0 * y])


# Optimizer base class
class Optimizer:
    def reset(self):
        pass


class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def reset(self):
        pass

    def step(self, grad):
        return -self.lr * grad


class Momentum(Optimizer):
    def __init__(self, lr, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def reset(self):
        self.v = np.zeros(2)

    def step(self, grad):
        self.v = self.momentum * self.v - self.lr * grad
        return self.v


class AdaGrad(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def reset(self):
        self.h = np.zeros(2)

    def step(self, grad):
        self.h += grad ** 2
        return -self.lr * grad / (np.sqrt(self.h) + 1e-7)


class RMSProp(Optimizer):
    def __init__(self, lr, beta=0.9):
        self.lr = lr
        self.beta = beta

    def reset(self):
        self.eg = np.zeros(2)

    def step(self, grad):
        self.eg = self.beta * self.eg + (1 - self.beta) * grad ** 2
        return -self.lr * grad / (np.sqrt(self.eg) + 1e-7)


class Adam(Optimizer):
    def __init__(self, lr, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

    def reset(self):
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0

    def step(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return -self.lr * m_hat / (np.sqrt(v_hat) + 1e-7)


# Grid for loss surface
x_vals = np.linspace(-10, 10, 400)
y_vals = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x_vals, y_vals)
Z = loss_fn(X, Y)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)
cset = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
fig.colorbar(cset, ax=ax)

# Init positions
init_pos = np.array([-7.0, 2.0])
max_steps = 200

# Create optimizers
optimizer_classes = {
    "SGD": lambda lr: SGD(lr),
    "Momentum": lambda lr: Momentum(lr),
    "AdaGrad": lambda lr: AdaGrad(lr),
    "RMSProp": lambda lr: RMSProp(lr),
    "Adam": lambda lr: Adam(lr),
}

colors = {"SGD": 'red', "Momentum": 'orange', "AdaGrad": 'cyan', "RMSProp": 'magenta', "Adam": 'lime'}
paths = {}
dots = {}
lines = {}

# Sliders
ax_lr = plt.axes([0.25, 0.15, 0.65, 0.03])
slider_lr = Slider(ax_lr, 'Learning Rate', 0.01, 1.5, valinit=0.1)

ax_epoch = plt.axes([0.25, 0.1, 0.65, 0.03])
slider_epoch = Slider(ax_epoch, 'Epoch', 1, max_steps, valinit=1, valstep=1)


# Run all optimizers and store paths
def run_all_optimizers(lr):
    new_paths = {}
    for name, cls in optimizer_classes.items():
        opt = cls(lr)
        if hasattr(opt, 'reset'):
            opt.reset()
        pos = init_pos.copy()
        history = [pos.copy()]
        for _ in range(max_steps):
            grad = grad_fn(*pos)
            pos += opt.step(grad)
            history.append(pos.copy())
        new_paths[name] = np.array(history)
    return new_paths


# Initial computation
paths = run_all_optimizers(slider_lr.val)

# Draw initial lines and points
for name, path in paths.items():
    line, = ax.plot([], [], label=name, lw=2, color=colors[name])
    point = ax.plot([], [], 'o', color=colors[name], markersize=5)[0]
    lines[name] = line
    dots[name] = point

ax.legend()


# Update function
def update(val):
    lr = slider_lr.val
    epoch = int(slider_epoch.val)

    global paths
    paths = run_all_optimizers(lr)

    for name in optimizer_classes:
        coords = paths[name]
        lines[name].set_data(coords[:epoch + 1, 0], coords[:epoch + 1, 1])
        dots[name].set_data([coords[epoch, 0]], [coords[epoch, 1]])

    fig.canvas.draw_idle()


slider_lr.on_changed(update)
slider_epoch.on_changed(update)

update(None)
plt.show()
