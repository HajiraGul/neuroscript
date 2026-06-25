import os
import joblib
import numpy as np
import tensorflow as tf

from config import *
from feature_extraction import (
    preprocess_image,
    extract_features_from_path
)


# =========================================================
# LOAD ALL MODELS
# =========================================================

def load_all_models():

    models = {}

    # -----------------------------------------------------

    if os.path.exists(RF_MODEL_PATH):

        models["Random Forest"] = joblib.load(
            RF_MODEL_PATH
        )

    # -----------------------------------------------------

    if os.path.exists(SVM_MODEL_PATH):

        models["SVM"] = joblib.load(
            SVM_MODEL_PATH
        )

    # -----------------------------------------------------

    if os.path.exists(CNN_MODEL_PATH):

        models["CNN"] = tf.keras.models.load_model(
            CNN_MODEL_PATH
        )

    # -----------------------------------------------------

    scaler = None

    if os.path.exists(SCALER_PATH):

        scaler = joblib.load(
            SCALER_PATH
        )

    # -----------------------------------------------------

    return models, scaler


# =========================================================
# PREDICT IMAGE
# =========================================================

def predict_image(
    image_path,
    selected_model="CNN"
):

    # -----------------------------------------------------

    models, scaler = load_all_models()

    # -----------------------------------------------------

    if selected_model not in models:

        raise ValueError(
            "Selected model is not trained yet. "
            "Run python train_models.py first."
        )

    # =====================================================
    # RANDOM FOREST / SVM
    # =====================================================

    if selected_model in [
        "Random Forest",
        "SVM"
    ]:

        if scaler is None:

            raise ValueError(
                "Scaler not found."
            )

        # -------------------------------------------------
        # EXTRACT FEATURES
        # -------------------------------------------------

        x = extract_features_from_path(
            image_path,
            IMAGE_SIZE
        )

        print("\nPrediction Features:")
        print(x)

        # -------------------------------------------------
        # RESHAPE
        # -------------------------------------------------

        x = x.reshape(1, -1)

        print("\nPrediction Feature Shape:")
        print(x.shape)

        # -------------------------------------------------
        # SCALE FEATURES
        # -------------------------------------------------

        x = scaler.transform(x)

        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        model = models[selected_model]

        pred = int(
            model.predict(x)[0]
        )

        probs = model.predict_proba(x)[0]

    # =====================================================
    # CNN
    # =====================================================

    else:

        img = preprocess_image(
            image_path,
            IMAGE_SIZE
        )

        img = img.reshape(
            1,
            IMAGE_SIZE,
            IMAGE_SIZE,
            1
        )

        probs = models["CNN"].predict(
            img,
            verbose=0
        )[0]

        pred = int(
            np.argmax(probs)
        )

    # =====================================================
    # RETURN RESULT
    # =====================================================

    return {

        "prediction_id": pred,

        "label": CLASS_NAMES[pred],

        "confidence": round(
            float(np.max(probs)) * 100,
            2
        ),

        "probabilities": [

            round(float(p) * 100, 2)

            for p in probs
        ]
    }
