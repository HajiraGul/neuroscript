NeuroScript Live Canvas Code

Main demo flow:
1. User writes on screen canvas using mouse/stylus/finger.
2. App shows live prediction while writing.
3. When user stops writing for a few seconds, app shows an auto-final prediction.
4. User can also click Predict Final Result for final output.
5. App shows SHAP, LIME, GenAI explanation, metrics, and model comparison.

Dataset folders:
dataset/normal
dataset/ocd_like
dataset/adhd_like

Run:
pip install -r requirements.txt
python train_models.py
streamlit run app.py

Important: This predicts handwriting pattern indicators only. It is not medical diagnosis.
