import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf


def reverse_preprocess_image(x, dtype=int):
    """Convert from preprocessed BGR [0-centered] to RGB [0-255]"""
    mean = [103.939, 116.779, 123.68]
    x = np.array(x)
    x[..., 0] += mean[0]
    x[..., 1] += mean[1]
    x[..., 2] += mean[2]
    x = x[..., ::-1]  # BGR -> RGB
    return np.array(x, dtype=dtype)


def clip_x_np(x):
    """For post-processing / logging: Tensor -> clipped RGB"""
    tmp = reverse_preprocess_image(np.array(x), dtype=float)
    return preprocess_image_np(np.clip(tmp, 0, 255))


def preprocess_image_np(x):
    """Standard VGG preprocessing using NumPy"""
    mean = [103.939, 116.779, 123.68]
    x = x[..., ::-1]  # RGB -> BGR
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    return x


def tf_clip_x(x):
    """Preprocess using TensorFlow ops to preserve gradients"""
    mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
    x_rev = x + mean[None, None, None, :]           # add mean
    x_rev = tf.reverse(x_rev, axis=[-1])            # BGR -> RGB
    x_rev = tf.clip_by_value(x_rev, 0.0, 255.0)      # clip
    x_proc = tf.reverse(x_rev, axis=[-1])            # RGB -> BGR
    x_proc = x_proc - mean[None, None, None, :]      # subtract mean
    return x_proc


def generate(neuron_selection, iterations=40, randm=False):
    model = VGG16(weights='imagenet')
    model_softmax = VGG16(weights='imagenet')

    # Disable softmax in the original model
    model.layers[-1].activation = None
    model = tf.keras.models.clone_model(model)
    model.set_weights(model_softmax.get_weights())

    # Initial image
    if randm:
        x = np.random.uniform(-8, 8, (224, 224, 3))
    else:
        x = np.zeros((224, 224, 3))

    x = tf.Variable(x[None, ...], dtype=tf.float32)  # Add batch dim
    optimizer = tf.keras.optimizers.Adam(learning_rate=2.0)

    for i in range(iterations):
        with tf.GradientTape() as tape:
            x_clipped = tf_clip_x(x)
            pred = model(x_clipped)
            loss = -pred[:, neuron_selection]

        G = tape.gradient(loss, x)
        if G is None:
            raise ValueError("Gradient is None. Graph might be broken.")

        optimizer.apply_gradients([(G, x)])

        # Logging
        x_np = x.numpy()[0]
        pred_value = model_softmax(np.array([clip_x_np(x_np)]))[-1][neuron_selection]
        print("i: {}, pred: {:.10f}".format(i, pred_value))

    # Visualize result
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(reverse_preprocess_image(np.array(x.numpy()[0])))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plot_path = f'../plots/gen_{"rand" if randm else "zeros"}_{neuron_selection}.png'
    plt.savefig(plot_path)
    plt.close()


if __name__ == '__main__':
    # generate(130, randm=True) # flamingo
    # generate(31, randm=True) # tree-frog
    # generate(953, randm=True) # pineapple
    # generate(980, randm=True) # volcano
    # generate(251, randm=True) # dalmatian
    # generate(278, randm=True) # kit fox
    # generate(9, randm=True) # ostrich
    generate(776, randm=True) # sax
