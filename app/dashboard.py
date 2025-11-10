# ============================================
# ✅ MUST CONFIGURE IMPORT PATH FIRST
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from src.model import load_model_artifacts
from src.agent import generate_investigation_summary
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ============================================
# ✅ NOW IMPORT MODULES SAFELY
# ============================================


def main():
    st.set_page_config(
        page_title="AI Fraud Detection",
        page_icon="🛡️",
        layout="wide"
    )

    st.title("🛡️ AI Fraud Detection Dashboard")
    st.write("Upload a dataset to generate fraud scores and AI investigation reports.")

    # ------------------------------------
    # ✅ FILE UPLOAD
    # ------------------------------------
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if not uploaded_file:
        st.info("⬆️ Upload `processed_fraud.csv` to continue.")
        return

    # ------------------------------------
    # ✅ LOAD DATA
    # ------------------------------------
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.dataframe(df.head())

    # ------------------------------------
    # ✅ LOAD MODEL + SHAP
    # ------------------------------------
    model, explainer = load_model_artifacts()

    # Remove label columns if present
    drop_cols = ["fraud", "Outcome", "label", "target"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Only numeric columns
    X = df.select_dtypes(include=[np.number]).copy()

    # ------------------------------------
    # ✅ PREDICTIONS
    # ------------------------------------
    fraud_prob = model.predict_proba(X)[:, 1]
    fraud_score = (fraud_prob * 100).round(2)

    df["fraud_prob"] = fraud_prob
    df["fraud_score"] = fraud_score

    df["risk_level"] = df["fraud_score"].apply(
        lambda x: "High Risk" if x >= 70 else
        ("Medium Risk" if x >= 40 else "Low Risk")
    )

    st.subheader("✅ Results")
    st.dataframe(df)

    # ------------------------------------
    # ✅ SHAP SUMMARY PLOT (CLOUD SAFE)
    # ------------------------------------
    st.subheader("🔍 SHAP Feature Importance (Cloud Safe)")

    shap_values = explainer.shap_values(X)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    st.pyplot(fig)

    # ------------------------------------
    # ✅ AI INVESTIGATION REPORT
    # ------------------------------------
    st.subheader("🧠 AI Investigation Report")

    report = generate_investigation_summary(
        X.iloc[0].to_dict(),
        shap_values[0],
        float(df["fraud_prob"].iloc[0]),
        float(df["fraud_score"].iloc[0]),
        str(df["risk_level"].iloc[0]),
    )

    st.text(report)


if __name__ == "__main__":
    main()
