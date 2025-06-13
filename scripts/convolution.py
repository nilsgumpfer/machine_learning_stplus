import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.interpolate import interp1d

# --- Parameters ---
f_coarse = np.array([0, 1, 2, 3, 2, 1, 0], dtype=float)
g_coarse = np.array([-1, 0, 1], dtype=float)
upsample_factor = 15

# --- Interpolate signal ---
t_f = np.linspace(0, len(f_coarse) - 1, len(f_coarse))
t_f_upsampled = np.linspace(0, len(f_coarse) - 1, len(f_coarse) * upsample_factor)
f = interp1d(t_f, f_coarse, kind='linear')(t_f_upsampled)
t = np.arange(len(f))

# --- Interpolate kernel ---
t_g = np.linspace(0, len(g_coarse) - 1, len(g_coarse))
t_g_upsampled = np.linspace(0, len(g_coarse) - 1, len(g_coarse) * upsample_factor)
g = interp1d(t_g, g_coarse, kind='linear')(t_g_upsampled)
g = g / np.sum(np.abs(g))  # Normalize for visualization

# --- Pad signal ---
pad_width = len(g) - 1
f_padded = np.pad(f, (pad_width, pad_width), mode='constant')
t_padded = np.arange(len(f_padded))

# --- Precompute convolution output (valid range only) ---
flipped_g = g[::-1]
output_len = len(f_padded) - len(flipped_g) + 1
conv_output_valid = np.array([
    np.sum(f_padded[i:i + len(g)] * flipped_g)
    for i in range(output_len)
])

# --- Center-shift convolution output ---
t_conv = t_padded[:output_len]
kernel_shift = len(g) // 2
t_conv_shifted = t_conv + kernel_shift

# Interpolate and stretch to full axis
conv_interp = interp1d(t_conv_shifted, conv_output_valid, kind='linear', fill_value="extrapolate")
conv_output = conv_interp(t_padded)

# --- Create figure and axes ---
fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
plt.subplots_adjust(bottom=0.25)

# --- Plot input signal ---
axs[0].set_title("Input Signal f(t)")
line_f, = axs[0].plot(t_padded, f_padded, label="f(t)", lw=2)

# --- Plot shifted kernel ---
axs[1].set_title("Shifted Kernel g(t - τ)")
kernel_line, = axs[1].plot(t_padded, np.zeros_like(f_padded), label="g(t - τ)", lw=2)

# --- Plot elementwise product ---
axs[2].set_title("Elementwise Product f(τ) · g(t - τ)")
product_line, = axs[2].plot(t_padded, np.zeros_like(f_padded), label="Product", lw=2)

# Dummy fill areas (updated dynamically)
product_area_positive = axs[2].fill_between(t_padded, 0, np.zeros_like(f_padded), color='green', alpha=0.3)
product_area_negative = axs[2].fill_between(t_padded, 0, np.zeros_like(f_padded), color='red', alpha=0.3)

# --- Plot convolution output (center-aligned) ---
axs[3].set_title("Convolution Output (f * g)(t)")
conv_line, = axs[3].plot(t_padded, conv_output, label="Convolution", lw=2)
sum_dot_conv, = axs[3].plot([], [], 'ro', label="Current Output")

for ax in axs:
    ax.grid(True)
    ax.legend(loc="upper right")

# --- Slider setup ---
max_slider_val = output_len - 1
ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
slider = Slider(ax_slider, 't (position)', 0, max_slider_val, valinit=0, valstep=1)

# --- Update function ---
def update(pos):
    global product_area_positive, product_area_negative
    pos = int(pos)
    shifted_g = np.zeros_like(f_padded)
    shifted_g[pos:pos + len(flipped_g)] = flipped_g

    product = f_padded * shifted_g
    conv_val = np.sum(product)

    kernel_line.set_ydata(shifted_g)
    product_line.set_ydata(product)
    sum_dot_conv.set_data([t_padded[pos + kernel_shift]], [conv_val])

    # Remove previous fills
    for coll in [product_area_positive, product_area_negative]:
        coll.remove()

    # Fill positive/negative areas under product
    product_area_positive = axs[2].fill_between(
        t_padded, 0, np.where(product > 0, product, 0),
        color='green', alpha=0.3
    )
    product_area_negative = axs[2].fill_between(
        t_padded, 0, np.where(product < 0, product, 0),
        color='red', alpha=0.3
    )

    fig.canvas.draw_idle()

slider.on_changed(update)
update(0)

plt.show()
