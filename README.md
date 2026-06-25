````markdown
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
| Machine Learning | Scikit-Learn |
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

# 📷 Screenshots

> Add your application screenshots inside an **images/** folder.

## 🏠 Home Screen

![Home](images/home.png)

---

## ✍️ Interactive Writing Canvas

![Canvas](images/canvas.png)

---

## ⚡ Real-Time Prediction

![Prediction](images/realtime_prediction.png)

---

## 📊 SHAP & LIME Explanation

![XAI](images/xai.png)

---

## 📄 AI Diagnostic Report

![Report](images/report.png)

---

# 🎥 Live Demo

Watch the complete project demonstration here.

▶️ **Demo Video**

https://your-demo-video-link.com

> Replace the above link with your YouTube, Google Drive, or LinkedIn demo video.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/neuroscript.git
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

---

# ⚠️ Disclaimer

NeuroScript is developed for **academic and research purposes only**.

The predictions generated by this system are intended to support preliminary neurological screening and **must not be considered a substitute for professional medical diagnosis or clinical evaluation.**

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

````
