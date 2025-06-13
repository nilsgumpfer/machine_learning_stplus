import os
import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers

# === PATHS ===
model_path = "../data/dino/model/dino_detector.h5"
image_path = "../data/dino/test_scene3.png"
base_path  = "../data/dino/patches"

# === CLASS MAP ===
def get_class_map(base_path):
    classes = sorted(os.listdir(base_path))
    return {cls_name: idx for idx, cls_name in enumerate(classes)}

# === AUGMENTATION ===
def augment_img(img):
    h, w, _ = img.shape
    crop_ratio = random.uniform(0.8, 1.0)
    new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
    y, x = random.randint(0, h - new_h), random.randint(0, w - new_w)
    img = img[y:y + new_h, x:x + new_w]

    if random.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if random.random() < 0.5:
        img = cv2.convertScaleAbs(img, alpha=1.0 + random.uniform(-0.1, 0.1), beta=random.uniform(-10, 10))
    return img

# === BALANCED DATASET ===
class BalancedFolderDataset(tf.keras.utils.Sequence):
    def __init__(self, base_path, class_map, input_size=(64, 64), batch_size=16, augment=True):
        self.samples = {cls: [] for cls in class_map}
        self.class_map = class_map
        self.input_size = input_size
        self.batch_size = batch_size
        self.augment = augment

        for cls in class_map:
            folder = os.path.join(base_path, cls)
            self.samples[cls] = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')]

        self.max_samples = max(len(lst) for lst in self.samples.values())
        self.all_samples = self._generate_balanced_samples()

    def _generate_balanced_samples(self):
        all_samples = []
        for cls_name, paths in self.samples.items():
            repeat = self.max_samples // len(paths)
            remainder = self.max_samples % len(paths)
            final_list = paths * repeat + random.sample(paths, remainder)
            all_samples.extend([(p, self.class_map[cls_name]) for p in final_list])
        random.shuffle(all_samples)
        return all_samples

    def __len__(self):
        return int(np.ceil(len(self.all_samples) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.all_samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []

        for img_path, class_idx in batch:
            img = cv2.imread(img_path)
            if self.augment:
                img = augment_img(img)
            img = cv2.resize(img, self.input_size)
            img = img.astype(np.float32) / 255.0

            one_hot_label = tf.keras.utils.to_categorical(class_idx, num_classes=len(self.class_map))
            X.append(img)
            y.append(one_hot_label)

        return np.array(X), np.array(y)

# === MODEL ===
def create_model(input_shape=(64, 64, 3), num_classes=3):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, 5, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 5, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 5, activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# === TRAIN ===
def train_model():
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    class_map = get_class_map(base_path)
    dataset = BalancedFolderDataset(base_path, class_map, augment=True)
    model = create_model(num_classes=len(class_map))
    model.compile(optimizer=optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(dataset, epochs=100)
    model.save(model_path)
    return model, class_map

# === HEATMAP INFERENCE ===
def predict_heatmaps(model_path, image_path, class_map, window=64, stride=16):
    model = tf.keras.models.load_model(model_path)
    inv_class_map = {v: k for k, v in class_map.items()}
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img_rgb.shape

    heatmaps = {cls: np.zeros(((h_img - window) // stride + 1, (w_img - window) // stride + 1))
                for cls in class_map}

    for i, y in enumerate(range(0, h_img - window + 1, stride)):
        for j, x in enumerate(range(0, w_img - window + 1, stride)):
            patch = img_rgb[y:y+window, x:x+window]
            patch_resized = cv2.resize(patch, (64, 64)).astype(np.float32) / 255.0
            probs = model.predict(np.expand_dims(patch_resized, axis=0), verbose=0)[0]

            for cls_id, prob in enumerate(probs):
                cls_name = inv_class_map[cls_id]
                heatmaps[cls_name][i, j] = prob

    for cls_name, heat in heatmaps.items():
        plt.figure()
        plt.imshow(heat, cmap='viridis', origin='upper')
        plt.title(f"Probability heatmap: {cls_name}")
        plt.colorbar()
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()

def draw_top_class_boxes(model_path, image_path, class_map, window=64, stride=16):
    model = tf.keras.models.load_model(model_path)
    inv_class_map = {v: k for k, v in class_map.items()}
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img_rgb.shape

    heatmaps = {cls: np.zeros(((h_img - window) // stride + 1, (w_img - window) // stride + 1))
                for cls in class_map}
    probs_map = {cls: np.zeros_like(heatmaps[cls]) for cls in class_map}

    # --- Slide window to collect class probabilities
    for i, y in enumerate(range(0, h_img - window + 1, stride)):
        for j, x in enumerate(range(0, w_img - window + 1, stride)):
            patch = img_rgb[y:y + window, x:x + window]
            patch_resized = cv2.resize(patch, (64, 64)).astype(np.float32) / 255.0
            probs = model.predict(np.expand_dims(patch_resized, axis=0), verbose=0)[0]

            for cls_id, prob in enumerate(probs):
                cls_name = inv_class_map[cls_id]
                heatmaps[cls_name][i, j] = prob
                probs_map[cls_name][i, j] = prob

    # --- Draw one bounding box per class (highest confidence)
    class_colors = {
        cls: tuple(np.random.randint(0, 255, 3).tolist())
        for cls in class_map
    }

    for cls_name, heatmap in heatmaps.items():
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        max_prob = heatmap[max_idx]
        i, j = max_idx
        y = i * stride
        x = j * stride
        color = class_colors[cls_name]

        # Draw box centered at this position
        cv2.rectangle(img_rgb, (x, y), (x + window, y + window), color, 2)
        label = f"{cls_name} ({max_prob:.2f})"
        cv2.putText(img_rgb, label, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    plt.figure(figsize=(12, 6))
    plt.imshow(img_rgb)
    plt.title("Top Bounding Box per Class")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()


# === MAIN ===
if __name__ == "__main__":
    model, class_map = train_model()
    predict_heatmaps(model_path, image_path, class_map)
    draw_top_class_boxes(model_path, image_path, class_map)
