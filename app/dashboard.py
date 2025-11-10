# ============================================================
# ✅ MUST BE FIRST — FIX IMPORT PATH BEFORE ANYTHING ELSE
# ============================================================
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd
import streamlit as st
from src.agent import generate_investigation_summary
from src.model import load_model_artifacts
import os
import sys

# Root = one level above /app → the project folder
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add root to Python path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# ✅ NOW IMPORT src MODULES (ONLY NOW — NOT BEFORE!)
# ============================================================

# ============================================================
# ✅ Standard imports
# ============================================================


def main():
    st.set_page_config(
        page_title="AI Fraud Detection",
        page_icon="🛡️",
        layout="wide"
    )

    st.title("🛡️ AI Fraud Detection Dashboard")
    st.write("Upload a dataset to generate fraud scores and reports.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if not uploaded_file:
        st.info("⬆️ Upload `processed_fraud.csv` to continue.")
        return

    # ✅ Load CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.dataframe(df.head())

    # ✅ Load model + explainer
    model, explainer = load_model_artifacts()

    # Remove labels if present
    label_cols = ["fraud", "Outcome", "label", "target"]
    df = df.drop(
        columns=[c for c in label_cols if c in df.columns], errors="ignore")

    # Only numeric features
    X = df.select_dtypes(include=[np.number])

    # ✅ Predictions
    fraud_prob = model.predict_proba(X)[:, 1]
    fraud_score = (fraud_prob * 100).round(2)

    df["fraud_prob"] = fraud_prob
    df["fraud_score"] = fraud_score
    df["risk_level"] = df["fraud_score"].apply(
        lambda x: "High Risk" if x >= 70 else
                  "Medium Risk" if x >= 40 else
                  "Low Risk"
    )

    st.subheader("✅ Results")
    st.dataframe(df)

    # ✅ SHAP Feature Importance (Cloud Safe)
    st.subheader("🔍 SHAP Feature Importance")

    shap_values = explainer.shap_values(X)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    st.pyplot(fig)

    # ✅ Investigation Agent
    st.subheader("🧠 AI Investigation Report")

    report = generate_investigation_summary(
        X.iloc[0].to_dict(),
        shap_values[0],
        float(df["fraud_prob"].iloc[0]),
        float(df["fraud_score"].iloc[0]),
        df["risk_level"].iloc[0]
    )

    st.text(report)


if __name__ == "__main__":
    main()
