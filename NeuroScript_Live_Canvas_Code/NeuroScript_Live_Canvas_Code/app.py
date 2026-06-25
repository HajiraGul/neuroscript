
import os
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from streamlit_drawable_canvas import st_canvas

from config import *
from predict_utils import predict_image, load_all_models
from xai_utils import explain_with_shap_single, explain_with_lime_single
from genai_utils import gemini_explanation


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NeuroScript Live Canvas",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(135deg, #050505, #101820, #0b0f14) !important;
        color: #ffffff !important;
    }

    div[data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #050505, #101820, #0b0f14) !important;
        color: #ffffff !important;
    }

    div[data-testid="stHeader"] {
        background-color: #050505 !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #334155;
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    .block-container {
        padding-top: 2rem;
        background: transparent !important;
        color: #ffffff !important;
    }

    h1 {
        color: #38bdf8 !important;
        font-weight: 800 !important;
        text-shadow: 0 0 12px rgba(56,189,248,0.4);
    }

    h2, h3 {
        color: #e0f2fe !important;
        font-weight: 700 !important;
    }

    h4, h5, h6, p, label, span, div {
        color: #f8fafc !important;
    }

    div[data-testid="stMarkdownContainer"] {
        color: #f8fafc !important;
    }

    iframe {
        background-color: #ffffff !important;
        border-radius: 14px !important;
    }

    canvas {
        background-color: #ffffff !important;
        border: 3px solid #38bdf8 !important;
        border-radius: 14px !important;
        box-shadow: 0 0 20px rgba(56,189,248,0.45);
    }

    div[data-testid="stMetric"] {
        background-color: #1e293b !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 14px !important;
        padding: 18px !important;
    }

    div[data-testid="stMetric"] * {
        color: #ffffff !important;
    }

    .stButton > button {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover {
        background-color: #0284c7 !important;
        color: #ffffff !important;
    }

    button[kind="primary"] {
        background: linear-gradient(90deg, #0284c7, #38bdf8) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 700 !important;
    }

    .canvas-card {
        background-color: #0f172a;
        padding: 18px;
        border-radius: 18px;
        border: 1px solid #334155;
        box-shadow: 0 0 25px rgba(15,23,42,0.9);
    }

    .info-card {
        background-color: #1e293b;
        padding: 18px;
        border-radius: 18px;
        border: 1px solid #38bdf8;
        box-shadow: 0 0 18px rgba(56,189,248,0.25);
        margin-bottom: 15px;
    }

    hr {
        border-color: #334155 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CREATE REQUIRED DIRECTORIES
# =========================================================

for d in [OUTPUT_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)


# =========================================================
# APP TITLE
# =========================================================

st.title("🧠 NeuroScript")
st.subheader("Live Handwriting Prediction with SHAP, LIME, and GenAI")


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Live Canvas Prediction",
        "Model Comparison",
        "XAI Results",
        "About Project"
    ]
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def has_writing(arr):

    if arr is None:
        return False

    rgb = arr[:, :, :3].astype(np.uint8)

    return np.mean(rgb < 245) > 0.002


def save_canvas(arr):

    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")

    path = os.path.join(TEMP_DIR, "live_canvas.png")

    img.save(path)

    return path, img


# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

def initialize_state():

    defaults = {

        "pred_history": [],
        "final_result": None,
        "final_image": None,
        "final_image_path": None,

        "shap_path": None,
        "lime_path": None,
        "top_features": [],
        "genai_text": None,

        "last_live_result": None,
        "last_canvas_hash": None,

        # IMPORTANT FIX
        "last_selected_model": None,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


# =========================================================
# GENERATE SHAP + LIME + GEMINI
# =========================================================

def generate_xai_and_genai(image_path, result):

    top_features = []
    shap_path = None
    lime_path = None

    shap_result = explain_with_shap_single(image_path)

    if shap_result:
        shap_path, top_features = shap_result

    models, _ = load_all_models()

    if "CNN" in models:
        lime_path = explain_with_lime_single(
            image_path,
            models["CNN"]
        )

    genai_text = gemini_explanation(
        result["label"],
        result["confidence"],
        top_features
    )

    st.session_state.shap_path = shap_path
    st.session_state.lime_path = lime_path
    st.session_state.top_features = top_features
    st.session_state.genai_text = genai_text


# =========================================================
# DISPLAY FINAL RESULT
# =========================================================

def display_final_result():

    if st.session_state.final_result is None:
        return

    result = st.session_state.final_result

    st.divider()

    st.header("Final Prediction")

    st.success(
        f"Prediction: {result['label']}"
    )

    st.info(
        f"Confidence: {result['confidence']}%"
    )

    # -----------------------------------------------------

    if st.session_state.final_image is not None:

        st.subheader("Captured Handwriting")

        st.image(
            st.session_state.final_image,
            use_container_width=True
        )

    # -----------------------------------------------------

    st.header("Explainable AI Result")

    # -----------------------------------------------------
    # SHAP
    # -----------------------------------------------------

    if (
        st.session_state.shap_path
        and os.path.exists(st.session_state.shap_path)
    ):

        st.subheader("SHAP Explanation")

        st.write(
            "SHAP shows which handwriting features influenced the prediction."
        )

        st.image(
            st.session_state.shap_path,
            use_container_width=True
        )

        if st.session_state.top_features:

            feature_df = pd.DataFrame(
                st.session_state.top_features,
                columns=[
                    "Handwriting Feature",
                    "Impact Value"
                ]
            )

            st.dataframe(
                feature_df,
                use_container_width=True
            )

    else:
        st.warning("SHAP explanation not available.")

    # -----------------------------------------------------
    # LIME
    # -----------------------------------------------------

    if (
        st.session_state.lime_path
        and os.path.exists(st.session_state.lime_path)
    ):

        st.subheader("LIME Explanation")

        st.write(
            "LIME highlights handwriting regions that influenced the CNN prediction."
        )

        st.image(
            st.session_state.lime_path,
            use_container_width=True
        )

    else:
        st.warning("LIME explanation not available.")

    # -----------------------------------------------------
    # GEMINI
    # -----------------------------------------------------

    st.subheader("🤖 Gemini AI Generated Explanation")

    if st.session_state.genai_text:

        st.write(st.session_state.genai_text)

    else:
        st.warning("GenAI explanation not available.")


# =========================================================
# INITIALIZE SESSION STATE
# =========================================================

initialize_state()


# =========================================================
# LIVE CANVAS PAGE
# =========================================================

if page == "Live Canvas Prediction":

    # -----------------------------------------------------
    # MODEL SELECT
    # -----------------------------------------------------

    selected_model = st.selectbox(
        "Prediction model",
        ["SVM", "Random Forest", "CNN"],
        index=0
    )

    stroke_width = st.slider(
        "Pen thickness",
        1,
        12,
        4
    )

    # -----------------------------------------------------
    # LAYOUT
    # -----------------------------------------------------

    col1, col2 = st.columns([2, 1])

    # =====================================================
    # LEFT SIDE - CANVAS
    # =====================================================

    with col1:

        st.markdown(
            '<div class="canvas-card">',
            unsafe_allow_html=True
        )

        canvas = st_canvas(
            fill_color="white",
            stroke_width=stroke_width,
            stroke_color="black",
            background_color="white",
            height=420,
            width=950,
            drawing_mode="freedraw",
            update_streamlit=True,
            key="canvas"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        st.caption(
            "Use the delete/trash icon below the canvas to clear the drawing."
        )

    # =====================================================
    # RIGHT SIDE - LIVE DETECTION
    # =====================================================

    with col2:

        st.markdown(
            '<div class="info-card">',
            unsafe_allow_html=True
        )

        st.subheader("Live Detection")

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        # -------------------------------------------------

        if (
            canvas.image_data is not None
            and has_writing(canvas.image_data)
        ):

            current_hash = hashlib.md5(
                canvas.image_data.tobytes()
            ).hexdigest()

            # =================================================
            # IMPORTANT FIX:
            # RE-PREDICT WHEN:
            # 1. CANVAS CHANGES
            # 2. MODEL CHANGES
            # =================================================

            if (
                current_hash != st.session_state.last_canvas_hash
                or selected_model != st.session_state.last_selected_model
            ):

                st.session_state.last_canvas_hash = current_hash
                st.session_state.last_selected_model = selected_model

                image_path, img = save_canvas(
                    canvas.image_data
                )

                try:

                    live_result = predict_image(
                        image_path,
                        selected_model
                    )

                    st.session_state.last_live_result = live_result

                    st.session_state.pred_history.append(
                        {
                            "Model": selected_model,
                            "Prediction": live_result["label"],
                            "Confidence": live_result["confidence"]
                        }
                    )

                    st.session_state.pred_history = (
                        st.session_state.pred_history[-10:]
                    )

                except Exception as e:

                    st.error(str(e))

                    st.info(
                        "Train models first using: python train_models.py"
                    )

            # -------------------------------------------------

            if st.session_state.last_live_result:

                live_result = st.session_state.last_live_result

                st.metric(
                    "Live Prediction",
                    live_result["label"],
                    f"{live_result['confidence']}%"
                )

                st.progress(
                    min(
                        live_result["confidence"] / 100,
                        1.0
                    )
                )

        else:

            st.info("Start writing on the canvas.")

    # =====================================================
    # BUTTONS
    # =====================================================

    st.divider()

    button_col1, button_col2 = st.columns(2)

    # -----------------------------------------------------
    # FINAL PREDICTION BUTTON
    # -----------------------------------------------------

    with button_col1:

        if st.button(
            "Predict Final Result",
            type="primary"
        ):

            if (
                canvas.image_data is not None
                and has_writing(canvas.image_data)
            ):

                image_path, img = save_canvas(
                    canvas.image_data
                )

                try:

                    final_result = predict_image(
                        image_path,
                        selected_model
                    )

                    st.session_state.final_result = final_result

                    st.session_state.final_image = img

                    st.session_state.final_image_path = image_path

                    generate_xai_and_genai(
                        image_path,
                        final_result
                    )

                    st.success(
                        "Final prediction generated successfully."
                    )

                except Exception as e:

                    st.error(str(e))

            else:

                st.warning(
                    "Please write something first."
                )

    # -----------------------------------------------------
    # CLEAR HISTORY BUTTON
    # -----------------------------------------------------

    with button_col2:

        if st.button("Clear Prediction History"):

            st.session_state.pred_history = []

            st.session_state.final_result = None
            st.session_state.final_image = None
            st.session_state.final_image_path = None

            st.session_state.shap_path = None
            st.session_state.lime_path = None
            st.session_state.top_features = []
            st.session_state.genai_text = None

            st.session_state.last_live_result = None
            st.session_state.last_canvas_hash = None

            # IMPORTANT FIX
            st.session_state.last_selected_model = None

            st.success(
                "Prediction history and final result cleared."
            )

    # =====================================================
    # SHOW FINAL RESULT
    # =====================================================

    display_final_result()


# =========================================================
# MODEL COMPARISON PAGE
# =========================================================

elif page == "Model Comparison":

    st.header("Model Comparison")

    if os.path.exists(METRICS_PATH):

        df = pd.read_csv(METRICS_PATH)

        st.dataframe(
            df,
            use_container_width=True
        )

        if "F1 Score" in df.columns:

            best_model = df.loc[
                df["F1 Score"].idxmax()
            ]

            st.success(
                f"Best Model: {best_model['Model']} "
                f"with F1 Score {best_model['F1 Score']}"
            )

        st.write(
            "The best model is selected using F1-score because it balances precision and recall."
        )

    else:

        st.warning(
            "Run python train_models.py first."
        )

    # -----------------------------------------------------

    if os.path.exists(CONFUSION_MATRIX_PATH):

        st.subheader("Confusion Matrix")

        st.image(
            CONFUSION_MATRIX_PATH,
            use_container_width=True
        )

    else:

        st.warning(
            "Confusion matrix not found. Run training again."
        )


# =========================================================
# XAI RESULTS PAGE
# =========================================================

elif page == "XAI Results":

    st.header("Stored XAI Outputs")

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # SHAP
    # -----------------------------------------------------

    with col1:

        st.subheader("SHAP Summary")

        if (
            st.session_state.shap_path
            and os.path.exists(st.session_state.shap_path)
        ):

            st.image(
                st.session_state.shap_path,
                use_container_width=True
            )

        elif os.path.exists(SHAP_PATH):

            st.image(
                SHAP_PATH,
                use_container_width=True
            )

        else:

            st.warning(
                "SHAP summary not found. Click Predict Final Result first."
            )

    # -----------------------------------------------------
    # LIME
    # -----------------------------------------------------

    with col2:

        st.subheader("LIME Example")

        if (
            st.session_state.lime_path
            and os.path.exists(st.session_state.lime_path)
        ):

            st.image(
                st.session_state.lime_path,
                use_container_width=True
            )

        elif os.path.exists(LIME_PATH):

            st.image(
                LIME_PATH,
                use_container_width=True
            )

        else:

            st.warning(
                "LIME output not found. Click Predict Final Result first."
            )


# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "About Project":

    st.header("About NeuroScript")

    st.write(
        """
        NeuroScript is an AI-based handwriting analysis system.

        The user writes on a live screen canvas.

        The system predicts one of three handwriting pattern classes:

        - Normal
        - OCD-like Handwriting Indicator
        - ADHD-like Handwriting Indicator

        The project includes:

        - Random Forest
        - SVM
        - CNN
        - Accuracy
        - Precision
        - Recall
        - F1-score
        - Model comparison
        - SHAP explanation
        - LIME explanation
        - Gemini AI generated explanation

        Important:
        This system does not medically diagnose OCD or ADHD.
        It only detects handwriting pattern indicators.
        """
    )