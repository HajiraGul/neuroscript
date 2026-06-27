

# 🧠 NeuroScript

> **AI-Based Handwriting Analysis for Real-Time Neurological Detection using Machine Learning, Explainable AI, and Generative AI.**

NeuroScript is an AI-powered healthcare research project designed to assist in the **early screening of neurological disorders** such as **Attention Deficit Hyperactivity Disorder (ADHD)** and **Obsessive-Compulsive Disorder (OCD)** through **real-time handwriting analysis**.

Unlike traditional handwriting analysis systems that require users to upload handwritten images, NeuroScript provides an **interactive digital canvas** where users can write naturally using a mouse, stylus, or touch-enabled device. The application captures handwriting strokes in real time, extracts meaningful handwriting features, predicts potential neurological conditions using Machine Learning models, explains every prediction through Explainable AI (SHAP & LIME), and generates an AI-powered diagnostic summary for better interpretability.

The system aims to provide a fast, non-invasive, transparent, and interactive solution for supporting early neurological screening.

---

# ✨ Features

- ✍️ Real-time handwriting input through an interactive canvas
- 🖱️ Supports mouse, stylus, and touch-based writing
- ⚡ Real-time handwriting capture and preprocessing
- 🧩 Automatic handwriting feature extraction
- 🤖 AI-powered neurological disorder prediction
- 🧠 Early ADHD screening
- 🧠 Early OCD screening
- 📊 Explainable AI using SHAP
- 🔍 Explainable AI using LIME
- 💬 AI-generated diagnostic summaries
- 📈 Interactive Streamlit dashboard
- 📂 Automatic report generation

---

# 🏗️ System Architecture

```text
          User Draws on Canvas
                   │
                   ▼
      Real-Time Handwriting Capture
                   │
                   ▼
      Handwriting Preprocessing
                   │
                   ▼
         Feature Extraction
                   │
                   ▼
      Machine Learning Model
                   │
                   ▼
          Disorder Prediction
                   │
                   ▼
 Explainable AI (SHAP & LIME)
                   │
                   ▼
     AI-Generated Diagnostic Report
                   │
                   ▼
       Interactive Streamlit Dashboard
```

---

# 🧠 Technologies Used

| Category | Technologies |
|-----------|-------------|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow |
| Data Processing | NumPy, Pandas |
| Computer Vision | OpenCV |
| Explainable AI | SHAP, LIME |
| Generative AI | OpenAI API |
| Web Framework | Streamlit |
| Visualization | Matplotlib |

---

# 📊 Dataset

The machine learning model is trained using a labeled handwriting dataset containing handwriting samples categorized into:

- ADHD
- OCD
- Normal

Instead of requiring users to upload handwriting images, NeuroScript captures handwriting **directly from an interactive digital canvas**. The captured stroke data is processed in real time, relevant handwriting features are extracted, and the trained model performs instant neurological screening.

---

# 🤖 Machine Learning

The system utilizes trained Machine Learning models to classify handwriting into one of the following categories:

- ADHD
- OCD
- Normal

The prediction process includes:

- Handwriting preprocessing
- Feature extraction
- Model inference
- Confidence score calculation

The best-performing trained model is integrated into the application for real-time prediction.

---

# 🔍 Explainable AI (XAI)

To ensure transparency and improve trust in AI predictions, NeuroScript integrates two Explainable AI techniques:

- **SHAP (SHapley Additive Explanations)**
- **LIME (Local Interpretable Model-Agnostic Explanations)**

These methods identify and visualize the handwriting characteristics that contributed most to the predicted neurological condition, making the model's decisions easier to understand for clinicians, researchers, and users.

---

# 💬 Generative AI

After generating a prediction, the application uses Generative AI to produce a human-readable diagnostic summary that includes:

- Prediction explanation
- Behavioral observations
- Clinical interpretation
- Recommendations
- Summary report

This bridges the gap between technical AI predictions and understandable healthcare insights.

---

# 📷 Screenshots<img width="1600" height="871" alt="WhatsApp Image 2026-06-25 at 3 00 21 PM (2)" src="https://github.com/user-attachments/assets/2227f86b-459e-4666-a26a-a8368dbb4dc0" />
<img width="1600" height="865" alt="WhatsApp Image 2026-06-25 at 2 34 14 PM" src="https://github.com/user-attachments/assets/cd300a98-fdcb-4f75-ab9f-37f0ce8d9592" />
<img width="1600" height="630" alt="WhatsApp Image 2026-06-25 at 2 34 15 PM" src="https://github.com/user-attachments/assets/36abd5bc-fdf7-4a83-8006-a177fb0b16a4" />
<img width="1600" height="861" alt="WhatsApp Image 2026-06-25 at 2 34 15 PM (1)" src="https://github.com/user-attachments/assets/dffd551b-bd9c-42d3-9836-e1d54dcd1220" />
<img width="1600" height="867" alt="WhatsApp Image 2026-06-25 at 2 34 15 PM (2)" src="https://github.com/user-attachments/assets/f65a2575-7007-4c93-9b86-8e5517d221ac" />
<img width="1600" height="856" alt="WhatsApp Image 2026-06-25 at 3 00 20 PM" src="https://github.com/user-attachments/assets/f0d5e3d8-7a3b-4cba-b2dd-a754678ad682" />
<img width="1600" height="864" alt="WhatsApp Image 2026-06-25 at 3 00 20 PM (1)" src="https://github.com/user-attachments/assets/5f98e6dc-861d-4e7b-83ec-dc5a30239c44" />
<img width="1600" height="865" alt="WhatsApp Image 2026-06-25 at 3 00 21 PM" src="https://github.com/user-attachments/assets/a36dc235-8045-450b-baea-4cfbf1e0f64f" />
<img width="1600" height="900" alt="WhatsApp Image 2026-06-25 at 3 00 21 PM (1)" src="https://github.com/user-attachments/assets/7d8fe1ed-9cc2-47a2-9886-1ad2500aaa3e" />
<img width="1600" height="871" alt="WhatsApp Image 2026-06-25 at 3 00 21 PM (2)" src="https://github.com/user-attachments/assets/81dba3c1-a746-4da0-9c38-0d6c7e604297" />

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/HajiraGul/neuroscript.git
```

Navigate to the project directory

```bash
cd neuroscript
```

Install all required dependencies

```bash
pip install -r requirements.txt
```

Launch the Streamlit application

```bash
streamlit run app.py
```

---

# 🚀 Usage

1. Launch the Streamlit application.
2. Open the interactive handwriting canvas.
3. Write naturally using your mouse, stylus, or touch-enabled device.
4. The application captures handwriting strokes in real time.
5. Handwriting features are extracted automatically.
6. The trained Machine Learning model predicts the neurological condition.
7. SHAP and LIME explain the prediction.
8. Generative AI produces a diagnostic summary.
9. The final prediction and report are displayed on the dashboard.

---

# 📂 Project Structure

```text
NeuroScript/
│
├── dataset/
│   └── Handwriting dataset used for training
│
├── outputs/
│   └── Generated reports, explanations and results
│
├── saved_models/
│   └── Trained machine learning models
│
├── temp_uploads/
│   └── Temporary canvas images
│
├── app.py
│   └── Main Streamlit application
│
├── config.py
│   └── Configuration settings
│
├── feature_extraction.py
│   └── Feature extraction module
│
├── genai_utils.py
│   └── AI report generation utilities
│
├── predict_utils.py
│   └── Prediction utilities
│
├── train_models.py
│   └── Model training script
│
├── xai_utils.py
│   └── SHAP & LIME implementation
│
├── requirements.txt
│
└── README.md
```

---

# 📊 Workflow

```text
User Writes on Interactive Canvas
               │
               ▼
     Capture Handwriting Strokes
               │
               ▼
     Real-Time Image Processing
               │
               ▼
        Feature Extraction
               │
               ▼
      Machine Learning Model
               │
               ▼
      Neurological Prediction
               │
               ▼
 Explainable AI (SHAP & LIME)
               │
               ▼
 AI-Generated Diagnostic Report
               │
               ▼
 Interactive Streamlit Dashboard
```

---

# 📁 Output

The application generates:

- ✅ Real-time neurological prediction
- 📈 Prediction confidence score
- 📊 SHAP explanation
- 🔍 LIME explanation
- 📄 AI-generated diagnostic summary
- 💡 Clinical recommendation report

Generated reports and visualizations are automatically saved in the **outputs/** directory.

---

# 🚀 Future Improvements

- Support additional neurological disorders
- Larger handwriting datasets
- Improved deep learning architectures
- Cloud deployment
- Mobile application integration
- Real-time stylus pressure and velocity analysis
- Multi-language handwriting support

---

# 📜 License

This project is licensed under the **MIT License**.


