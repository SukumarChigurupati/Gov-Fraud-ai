import json


def generate_investigation_summary(feature_values, shap_values, fraud_prob, fraud_score, risk_level):
    """
    Creates an AI-style fraud investigation summary.
    """

    # Pair feature names with their SHAP values
    shap_pairs = list(zip(feature_values.keys(), shap_values))
    shap_sorted = sorted(shap_pairs, key=lambda x: abs(x[1]), reverse=True)
    top_features = shap_sorted[:3]

    summary = "🔎 FRAUD INVESTIGATION REPORT\n\n"
    summary += f"Fraud Probability: {fraud_prob:.3f}\n"
    summary += f"Fraud Score: {fraud_score}/100\n"
    summary += f"Risk Level: {risk_level}\n\n"

    summary += "Top Contributing Features:\n"
    for feat, val in top_features:
        summary += f"- {feat}: SHAP impact = {val:.4f}\n"

    summary += "\nFeature Values:\n"
    for feat, val in feature_values.items():
        summary += f"- {feat}: {val}\n"

    summary += "\nInterpretation:\n"
    summary += (
        f"The model classified this transaction as **{risk_level}** based on deviations "
        "from typical patterns in the dataset. The top contributing features indicate "
        "why the model increased or decreased the fraud probability.\n"
    )

    summary += "\nRecommendation:\n"
    if risk_level == "High Risk":
        summary += "Immediate manual review recommended.\n"
    elif risk_level == "Medium Risk":
        summary += "Review recommended if additional red flags exist.\n"
    else:
        summary += "No manual review needed.\n"

    return summary
