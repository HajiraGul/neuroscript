import os
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

from lime import lime_image
from skimage.segmentation import mark_boundaries

from config import *
from feature_extraction import preprocess_image, extract_features_from_path


def explain_with_shap_single(image_path):
    """
    SHAP explanation using Random Forest.
    Shows which handwriting features affected the prediction.
    """

    if not os.path.exists(RF_MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None

    try:
        rf = joblib.load(RF_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        features = extract_features_from_path(image_path, IMAGE_SIZE).reshape(1, -1)
        features_scaled = scaler.transform(features)

        pred = int(rf.predict(features_scaled)[0])

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(features_scaled)

        if isinstance(shap_values, list):
            values = shap_values[pred][0]
        else:
            values = np.array(shap_values).reshape(-1)[:len(FEATURE_NAMES)]

        output_path = os.path.join(OUTPUT_DIR, "shap_uploaded.png")

        plt.figure(figsize=(9, 5))
        plt.barh(FEATURE_NAMES, values)
        plt.title(f"SHAP Explanation for {CLASS_NAMES[pred]}")
        plt.xlabel("Feature Impact on Prediction")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        top_features = sorted(
            zip(FEATURE_NAMES, values),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        return output_path, top_features

    except Exception as e:
        print("SHAP explanation failed:", e)
        return None


def explain_with_lime_single(image_path, cnn_model):
    """
    LIME explanation using CNN.
    Highlights the handwriting image regions that affected the prediction.
    """

    try:
        img = preprocess_image(image_path, IMAGE_SIZE)
        rgb_img = np.stack([img, img, img], axis=-1)

        def predict_fn(images):
            batch = []

            for image in images:
                gray = image[:, :, 0]
                batch.append(gray.reshape(IMAGE_SIZE, IMAGE_SIZE, 1))

            return cnn_model.predict(np.array(batch), verbose=0)

        explainer = lime_image.LimeImageExplainer()

        explanation = explainer.explain_instance(
            rgb_img,
            predict_fn,
            top_labels=NUM_CLASSES,
            hide_color=0,
            num_samples=500
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=8,
            hide_rest=False
        )

        output_path = os.path.join(OUTPUT_DIR, "lime_uploaded.png")

        plt.figure(figsize=(6, 6))
        plt.imshow(mark_boundaries(temp, mask))
        plt.axis("off")
        plt.title("LIME Highlighted Handwriting Regions")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        return output_path

    except Exception as e:
        print("LIME explanation failed:", e)
        return None