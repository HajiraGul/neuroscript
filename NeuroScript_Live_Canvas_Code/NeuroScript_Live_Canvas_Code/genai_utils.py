import os
from dotenv import load_dotenv
load_dotenv()

def offline_genai_explanation(prediction, confidence, top_features):
    feats=', '.join([str(f[0]) for f in top_features]) if top_features else 'spacing, alignment, stroke density, and writing structure'
    return f"Prediction: {prediction}\nConfidence: {confidence}%\n\nThe model considered handwriting features such as {feats}. This is an AI-based handwriting pattern indicator only, not a medical diagnosis."

def gemini_explanation(prediction, confidence, top_features):
    api_key=os.getenv('GEMINI_API_KEY')
    if not api_key:
        return offline_genai_explanation(prediction,confidence,top_features)
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        feats=', '.join([str(f[0]) for f in top_features]) if top_features else 'not available'
        prompt=f'Explain this AI handwriting result simply for a university demo. Prediction: {prediction}. Confidence: {confidence}%. Features: {feats}. Mention this is not medical diagnosis.'
        return genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt).text
    except Exception:
        return offline_genai_explanation(prediction,confidence,top_features)
