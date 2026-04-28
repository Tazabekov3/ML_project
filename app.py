import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ── Load model and metadata ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_metadata():
    with open("feature_names.json") as f:
        features = json.load(f)
    with open("sample_input.json") as f:
        sample = json.load(f)
    return features, sample

pipeline = load_model()
feature_names, sample_input = load_metadata()

# ── UI ───────────────────────────────────────────────────────────────────
st.title("Sleep Anomaly Detector")
st.markdown("Detects anomalous sleep epochs from EEG features using a trained Random Forest pipeline.")

st.sidebar.header("Input EEG Features")
st.sidebar.markdown("Adjust values or use the pre-loaded sample epoch.")

user_input = {}
for feat in feature_names:
    default = float(sample_input.get(feat, 0.0))
    user_input[feat] = st.sidebar.number_input(
        feat, value=default, format="%.5f"
    )

if st.button("Predict & Explain"):
    X_input = pd.DataFrame([user_input])

    # ── Prediction ──────────────────────────────────────────────────────
    prediction = pipeline.predict(X_input)[0]
    proba = pipeline.predict_proba(X_input)[0]

    label = "Anomalous" if prediction == 1 else "Normal"
    confidence = proba[prediction]

    st.subheader("Prediction")
    st.metric("Result", label, f"Confidence: {confidence:.1%}")
    st.progress(float(proba[1]), text=f"Anomaly probability: {proba[1]:.1%}")

    # ── SHAP Explanation ─────────────────────────────────────────────────
    st.subheader("SHAP Explanation")
    st.markdown("Which features pushed this prediction toward Anomalous or Normal?")

    # Transform input through all pipeline steps except final estimator
    preprocessor_steps = pipeline[:-1]
    X_transformed = preprocessor_steps.transform(X_input)

    rf_model = pipeline.named_steps["estimator"]
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_transformed)

    # Get selected feature names after SelectKBest
    selector = pipeline.named_steps["feature_selector"]
    selected_indices = selector.get_support(indices=True)
    selected_feat_names = [feature_names[i] for i in selected_indices]

    # Handle both SHAP output formats:
    # Old: shap_values is a list [class_0_array, class_1_array]
    # New: shap_values is a single 3D array of shape (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]   # class 1, first sample
    else:
        sv = shap_values[0, :, 1] if shap_values.ndim == 3 else shap_values[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.bar_plot(
        sv,
        feature_names=selected_feat_names,
        max_display=10,
        show=False
    )
    plt.title("Top 10 Features Contributing to Anomaly Score")
    plt.tight_layout()
    st.pyplot(fig)