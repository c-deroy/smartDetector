"""
Streamlit Web Application for the current PaySim fraud detection system.
"""
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Smart Detector | PaySim Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"
TYPE_OPTIONS = ["PAYMENT", "CASH_OUT", "DEBIT", "TRANSFER"]
MERCHANT_CAT_OPTIONS = ["grocery", "restaurant", "retail", "travel", "utilities"]
COUNTRY_OPTIONS = ["CN", "DE", "FR", "GB", "IN", "JP", "NG", "RU", "US"]


def check_api_status():
    """Check whether the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def build_named_feature_payload(raw_row):
    """Convert a UI row or CSV row into the feature dict expected by /predict_named."""
    if not isinstance(raw_row, dict):
        raise ValueError("Feature payload must be a dictionary.")

    cleaned = {k: (0 if v is None else v) for k, v in raw_row.items()}
    payload = {
        "step": int(cleaned.get("step", 1)),
        "time_hour": int(cleaned.get("time_hour", 12)),
        "amount": float(cleaned.get("amount", 0.0)),
        "oldbalanceOrg": float(cleaned.get("oldbalanceOrg", 0.0)),
        "newbalanceOrig": float(cleaned.get("newbalanceOrig", 0.0)),
        "oldbalanceDest": float(cleaned.get("oldbalanceDest", 0.0)),
        "newbalanceDest": float(cleaned.get("newbalanceDest", 0.0)),
        "orig_amount_mean": float(cleaned.get("orig_amount_mean", 0.0)),
        "orig_amount_std": float(cleaned.get("orig_amount_std", 0.0)),
        "time_since_prev": float(cleaned.get("time_since_prev", 0.0)),
        "tx_count_24h": int(cleaned.get("tx_count_24h", 0)),
        "amount_z": float(cleaned.get("amount_z", 0.0)),
        "is_unusual_amount": int(cleaned.get("is_unusual_amount", 0)),
        "merchant_risk": float(cleaned.get("merchant_risk", 0.0)),
        "account_age_days": int(cleaned.get("account_age_days", 365)),
        "auth_failures": int(cleaned.get("auth_failures", 0)),
    }

    tx_type = str(cleaned.get("type", cleaned.get("transaction_type", "PAYMENT")).upper())
    merchant_cat = str(cleaned.get("merchant_cat", "retail")).lower()
    country = str(cleaned.get("country", "US")).upper()

    for item in TYPE_OPTIONS:
        payload[f"type_{item}"] = 1.0 if tx_type == item else 0.0

    for item in MERCHANT_CAT_OPTIONS:
        payload[f"merchant_cat_{item}"] = 1.0 if merchant_cat == item else 0.0

    for item in COUNTRY_OPTIONS:
        payload[f"country_{item}"] = 1.0 if country == item else 0.0

    for key in ["type_PAYMENT", "type_CASH_OUT", "type_DEBIT", "type_TRANSFER"]:
        if key in cleaned:
            payload[key] = float(cleaned[key])
    for key in ["merchant_cat_grocery", "merchant_cat_restaurant", "merchant_cat_retail", "merchant_cat_travel", "merchant_cat_utilities"]:
        if key in cleaned:
            payload[key] = float(cleaned[key])
    for key in ["country_CN", "country_DE", "country_FR", "country_GB", "country_IN", "country_JP", "country_NG", "country_RU", "country_US"]:
        if key in cleaned:
            payload[key] = float(cleaned[key])

    return payload


def call_predict_named(payload):
    response = requests.post(f"{API_URL}/predict_named", json=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def single_transaction_page():
    """Single transaction fraud detection page using the current artifact model."""
    st.header("Single Transaction Fraud Detection")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Transaction Profile")
        with st.form("transaction_form"):
            left_col, right_col = st.columns(2)

            with left_col:
                amount = st.number_input("amount", min_value=0.0, value=245.0, step=1.0)
                step = st.number_input("step", min_value=1, value=26, step=1)
                time_hour = st.number_input("time_hour", min_value=0, max_value=23, value=21, step=1)
                oldbalanceOrg = st.number_input("oldbalanceOrg", min_value=0.0, value=5000.0, step=10.0)
                newbalanceOrig = st.number_input("newbalanceOrig", min_value=0.0, value=4700.0, step=10.0)
                oldbalanceDest = st.number_input("oldbalanceDest", min_value=0.0, value=0.0, step=10.0)
                newbalanceDest = st.number_input("newbalanceDest", min_value=0.0, value=0.0, step=10.0)
                time_since_prev = st.number_input("time_since_prev", min_value=0.0, value=5.0, step=0.5)
                tx_count_24h = st.number_input("tx_count_24h", min_value=0, value=2, step=1)
                amount_z = st.number_input("amount_z", min_value=0.0, value=1.2, step=0.1)

            with right_col:
                merchant_risk = st.slider("merchant_risk", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
                account_age_days = st.number_input("account_age_days", min_value=1, value=365, step=1)
                auth_failures = st.number_input("auth_failures", min_value=0, value=0, step=1)
                is_unusual_amount = st.checkbox("is_unusual_amount", value=True)
                transaction_type = st.selectbox("type", TYPE_OPTIONS, index=0)
                merchant_cat = st.selectbox("merchant_cat", MERCHANT_CAT_OPTIONS, index=2)
                country = st.selectbox("country", COUNTRY_OPTIONS, index=8)
                orig_amount_mean = st.number_input("orig_amount_mean", min_value=0.0, value=180.0, step=10.0)
                orig_amount_std = st.number_input("orig_amount_std", min_value=0.0, value=45.0, step=1.0)

            submitted = st.form_submit_button("🔍 Check Fraud Risk")

    with col2:
        st.subheader("Prediction")
        if submitted and check_api_status():
            payload = build_named_feature_payload({
                "step": step,
                "time_hour": time_hour,
                "amount": amount,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest,
                "orig_amount_mean": orig_amount_mean,
                "orig_amount_std": orig_amount_std,
                "time_since_prev": time_since_prev,
                "tx_count_24h": tx_count_24h,
                "amount_z": amount_z,
                "is_unusual_amount": int(is_unusual_amount),
                "merchant_risk": merchant_risk,
                "account_age_days": account_age_days,
                "auth_failures": auth_failures,
                "type": transaction_type,
                "merchant_cat": merchant_cat,
                "country": country,
            })

            try:
                result = call_predict_named(payload)
                prob = float(result.get("fraud_probability", 0.0))
                risk_level = result.get("risk_level", "Very Low Risk")
                is_fraud = bool(result.get("is_fraudulent", False))

                if risk_level == "High Risk":
                    icon = "🔴"
                elif risk_level == "Medium Risk":
                    icon = "🟠"
                elif risk_level == "Low Risk":
                    icon = "🟡"
                else:
                    icon = "🟢"

                st.metric("Fraud Probability", f"{prob:.1%}")
                st.metric("Risk Level", f"{icon} {risk_level}")

                if is_fraud:
                    st.error("🚨 Potential Fraud")
                else:
                    st.success("✅ Normal Transaction")

                explanation = result.get("explanation", {}) or {}
                if explanation.get("reason"):
                    st.write(explanation["reason"])

                feature_impacts = explanation.get("feature_impacts", []) or []
                if feature_impacts:
                    top_df = pd.DataFrame(feature_impacts).head(5)
                    top_df = top_df[["feature", "contribution", "direction"]]
                    st.dataframe(top_df, use_container_width=True)

                    fig, ax = plt.subplots(figsize=(8, 4))
                    values = top_df["contribution"].astype(float).abs().tolist()
                    labels = top_df["feature"].tolist()
                    ax.barh(labels[::-1], values[::-1], color=["#ef4444" if x > 0 else "#22c55e" for x in top_df["contribution"].astype(float).tolist()][::-1])
                    ax.set_title("Top Feature Contributions")
                    ax.set_xlabel("Absolute contribution")
                    plt.tight_layout()
                    st.pyplot(fig)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
        elif submitted:
            st.error("API not connected. Please start the backend.")


def batch_analysis_page():
    """Batch analysis page for CSV uploads matching the current feature schema."""
    st.header("Batch Fraud Analysis")
    st.write("Upload a CSV containing PaySim-style transaction columns such as amount, time_hour, type, merchant_cat, country, merchant_risk, tx_count_24h, and other engineered fields.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())

            if st.button("Analyze uploaded file"):
                if not check_api_status():
                    st.error("API not connected.")
                    return

                with st.spinner("Running model inference across the batch..."):
                    results = []
                    for _, row in df.iterrows():
                        try:
                            payload = build_named_feature_payload(row.to_dict())
                            result = call_predict_named(payload)
                            results.append({
                                "fraud_probability": float(result.get("fraud_probability", 0.0)),
                                "is_fraudulent": bool(result.get("is_fraudulent", False)),
                                "risk_level": result.get("risk_level", "Very Low Risk"),
                            })
                        except Exception:
                            results.append({
                                "fraud_probability": 0.0,
                                "is_fraudulent": False,
                                "risk_level": "Unavailable",
                            })

                    result_df = pd.DataFrame(results)
                    merged = pd.concat([df.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)
                    st.dataframe(merged.head(50), use_container_width=True)

                    counts = result_df["is_fraudulent"].value_counts().to_dict()
                    fraud_count = counts.get(True, 0)
                    total_count = len(result_df)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Rows", total_count)
                    with col2:
                        st.metric("Fraud Detected", fraud_count)
                    with col3:
                        st.metric("Legitimate", total_count - fraud_count)

                    probs = result_df["fraud_probability"].tolist()
                    fig, ax = plt.subplots()
                    ax.hist(probs, bins=20, range=(0, 1), alpha=0.8)
                    ax.set_title("Fraud probability distribution")
                    ax.set_xlabel("Probability")
                    ax.set_ylabel("Count")
                    st.pyplot(fig)

        except Exception as exc:
            st.error(f"Error processing CSV: {exc}")


def model_performance_page():
    """Model information page for the current artifact-based deployment."""
    st.header("Model Performance & Deployment Status")

    if not check_api_status():
        st.error("API not connected. Start the backend first.")
        return

    try:
        info = requests.get(f"{API_URL}/model-info", timeout=15).json()
        st.subheader("Loaded artifacts")
        st.write(info)

        available = info.get("available_models", [])
        if available:
            st.success(f"Model backend is active: {', '.join(available)}")

        st.markdown("""
        This deployment uses the trained model artifacts saved from the PaySim pipeline, not the legacy credit-card PCA flow.
        The current model input follows the engineered feature schema exported in metadata and mapped to named features at inference time.
        """)

        status_cols = st.columns(2)
        with status_cols[0]:
            st.metric("Available Models", len(available))
        with status_cols[1]:
            st.metric("Feature Selection", info.get("features_selected", 0))

    except Exception as exc:
        st.error(f"Unable to fetch model info: {exc}")


def architecture_page():
    """System architecture explanation for the current model pipeline."""
    st.header("Current Fraud Detection Architecture")

    st.markdown("""
    ## 1. Data layer
    The project now uses engineered PaySim transaction features instead of the older PCA-based credit-card schema.
    The canonical feature order is stored in the model metadata and used during inference to map named inputs into the trained model layout.

    ## 2. Feature engineering
    Derived fields include:
    - amount deviation and unusual-amount flags
    - time_since_prev and tx_count_24h
    - merchant_risk and account_age_days
    - auth_failures and customer timing signals
    - encoded transaction type, merchant category, and country flags

    ## 3. Model layer
    The backend loads trained artifacts from the models folder and uses the named-feature route for production scoring.
    This keeps the UI and backend aligned with the exact model input shape needed for prediction.

    ## 4. Explainability layer
    SHAP-based explanations are generated from the loaded model and returned as feature_impacts with a human-readable reason so the dashboard can explain the decision in plain language.
    """)

    st.subheader("Model flow")
    st.code(
        "User form / CSV -> build_named_feature_payload() -> /predict_named -> model artifact -> explanation -> dashboard",
        language="text",
    )

    st.subheader("Current feature groups")
    st.code(
        "amount, time_hour, merchant_risk, tx_count_24h, type_*, merchant_cat_*, country_*, auth_failures, is_unusual_amount",
        language="text",
    )


def main():
    st.title("Smart Detector")
    st.caption("PaySim fraud detection system with artifact-based inference and SHAP explanations")

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Choose a page:", [
        "Single Transaction Check",
        "Batch Analysis",
        "Model Performance",
        "System Architecture",
    ])

    api_status = check_api_status()
    if api_status:
        st.sidebar.success("✅ API connected")
    else:
        st.sidebar.error("❌ API disconnected")
        st.warning("Start the backend with: uvicorn api:app --host 0.0.0.0 --port 8000")

    if page == "Single Transaction Check":
        single_transaction_page()
    elif page == "Batch Analysis":
        batch_analysis_page()
    elif page == "Model Performance":
        model_performance_page()
    else:
        architecture_page()


if __name__ == "__main__":
    main()
