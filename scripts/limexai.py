import matplotlib.pyplot as plt
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.applications import vgg16, resnet50, resnet, imagenet_utils
from tensorflow.keras.preprocessing import image
import cv2


def load_model_and_preprocess(model_name):
    model_name = model_name.lower()
    if model_name == 'vgg16':
        model = vgg16.VGG16(weights='imagenet')
        preprocess = vgg16.preprocess_input
        input_size = (224, 224)
    elif model_name == 'resnet50':
        model = resnet50.ResNet50(weights='imagenet')
        preprocess = resnet50.preprocess_input
        input_size = (224, 224)
    elif model_name == 'resnet101':
        model = resnet.ResNet101(weights='imagenet')
        preprocess = resnet50.preprocess_input
        input_size = (224, 224)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model, preprocess, input_size


def load_image(img_path, input_size):
    img = image.load_img(img_path, target_size=input_size)
    img_array = image.img_to_array(img)
    raw_copy = img_array.copy().astype(np.uint8)
    return raw_copy


def get_lime_explanation(img_array, model, preprocess_fn, top_labels=1, num_samples=1000):
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        images = preprocess_fn(images.copy().astype(np.float32))
        return model.predict(images)

    explanation = explainer.explain_instance(
        img_array,
        classifier_fn=predict_fn,
        top_labels=top_labels,
        hide_color=0,
        num_samples=num_samples
    )

    return explanation


def show_lime_result(img_array, explanation, label_idx):
    if img_array.dtype != np.uint8:
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    segmentation = explanation.segments
    weights = dict(explanation.local_exp[label_idx])  # {segment_id: weight}
    heatmap = np.zeros_like(segmentation, dtype=np.float32)

    for seg_val, weight in weights.items():
        heatmap[segmentation == seg_val] = weight

    # Normalize heatmap to [0, 1] for visualization
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Convert to 3-channel heatmap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on original image
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_color, 0.4, 0)

    # Plot both
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_array)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("LIME Heatmap")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.show()



def run_lime_xai(
    img_path,
    model_name='vgg16',
    top_labels=1,
    num_samples=100
):
    model, preprocess, input_size = load_model_and_preprocess(model_name)
    img_array = load_image(img_path, input_size)

    explanation = get_lime_explanation(img_array, model, preprocess, top_labels, num_samples)

    # Predict class to decode label
    input_tensor = np.expand_dims(preprocess(img_array.copy().astype(np.float32)), axis=0)
    preds = model.predict(input_tensor, verbose=0)
    decoded = imagenet_utils.decode_predictions(preds, top=1)[0][0]
    label_idx = np.argmax(preds[0])

    print(f"Predicted class: {decoded[1]} ({decoded[2]*100:.2f}%)")

    show_lime_result(img_array, explanation, label_idx)


if __name__ == '__main__':
    run_lime_xai(
        img_path="../data/cat.jpg",
        model_name="resnet50",
        top_labels=1,
        num_samples=1000
    )
