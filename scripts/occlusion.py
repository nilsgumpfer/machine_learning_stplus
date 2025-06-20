import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import vgg16, resnet50, resnet, imagenet_utils
from tensorflow.keras.preprocessing import image


def load_model_and_preprocess(model_name):
    model_name = model_name.lower()
    if model_name == 'vgg16':
        model = vgg16.VGG16(weights='imagenet')
        preprocess = vgg16.preprocess_input
        size = (224, 224)
    elif model_name == 'resnet50':
        model = resnet50.ResNet50(weights='imagenet')
        preprocess = resnet50.preprocess_input
        size = (224, 224)
    elif model_name == 'resnet101':
        model = resnet.ResNet101(weights='imagenet')
        preprocess = resnet50.preprocess_input  # same preprocessing
        size = (224, 224)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model, preprocess, size


def load_and_preprocess_image(img_path, target_size, preprocess):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    return preprocess(np.array(img_batch)), img_array


def occlusion_heatmap(model, img, patch_size=20, stride=10, occlusion_value=0.0, class_index=None):
    H, W, _ = img.shape
    heatmap_h = (H - patch_size) // stride + 1
    heatmap_w = (W - patch_size) // stride + 1
    heatmap = np.zeros((heatmap_h, heatmap_w))
    original_pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]

    if class_index is None:
        class_index = np.argmax(original_pred)

    for hi, i in enumerate(range(0, H - patch_size + 1, stride)):
        for hj, j in enumerate(range(0, W - patch_size + 1, stride)):
            occluded = img.copy()
            occluded[i:i + patch_size, j:j + patch_size, :] = occlusion_value
            pred = model.predict(np.expand_dims(occluded, axis=0), verbose=0)[0]
            prob_drop = original_pred[class_index] - pred[class_index]
            heatmap[hi, hj] = prob_drop

    return heatmap, class_index



def visualize_occlusion(img_array, heatmap):
    # Ensure input image is in uint8 format (0–255)
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap_normalized = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized) + 1e-8)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_normalized), cv2.COLORMAP_JET)

    # OpenCV outputs BGR; convert to RGB for matplotlib
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay: weighted blend of heatmap and original image
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_rgb, 0.4, 0)

    # Display with matplotlib (expects RGB, uint8 or float in [0,1])
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_array)  # already in RGB format
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Occlusion Heatmap")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.show()




def run_occlusion_xai(
    img_path,
    model_name='vgg16',
    patch_size=20,
    stride=10,
    use_gray=False
):
    model, preprocess, input_size = load_model_and_preprocess(model_name)
    input_tensor, raw_img = load_and_preprocess_image(img_path, input_size, preprocess)

    occlusion_value = 127.0 if use_gray else 0.0   # use black or gray
    heatmap, class_idx = occlusion_heatmap(model, input_tensor[0], patch_size, stride, occlusion_value)

    pred = model.predict(input_tensor, verbose=0)
    label, name, confidence = imagenet_utils.decode_predictions(pred, top=1)[0][0]
    print(f"Predicted class: {name} (ID: {label}) – Confidence: {confidence:.2f}")

    visualize_occlusion(raw_img, heatmap)


if __name__ == '__main__':
    run_occlusion_xai(
        img_path="../data/cat.jpg",    # your image path
        model_name="resnet50",         # 'vgg16', 'resnet50', 'resnet101'
        patch_size=30,
        stride=15,
        use_gray=True                  # use False for black patches
    )
