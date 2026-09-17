"""
Streamlit Web Application for Credit Card Fraud Detection System
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = "http://localhost:8000"

MODEL_FEATURE_NAMES = [
    "step", "time_hour", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "orig_amount_mean", "orig_amount_std",
    "time_since_prev", "tx_count_24h", "amount_z", "is_unusual_amount",
    "merchant_risk", "account_age_days", "auth_failures", "device_account_count",
    "ip_account_count", "beneficiary_tx_count", "merchant_tx_count", "is_new_device",
    "is_new_ip", "is_new_beneficiary", "is_new_merchant", "rapid_withdrawal_count",
    "customer_category_deviation", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT",
    "type_TRANSFER", "merchant_cat_grocery", "merchant_cat_restaurant",
    "merchant_cat_retail", "merchant_cat_travel", "merchant_cat_utilities",
    "country_CN", "country_DE", "country_FR", "country_GB", "country_IN",
    "country_JP", "country_NG", "country_RU", "country_US", "isFlaggedFraud_1",
    "transaction_channel_BANK_TRANSFER", "transaction_channel_ONLINE",
    "transaction_channel_POS"
]

FEATURE_LABELS = {
    "amount": "Transaction amount",
    "oldbalanceOrg": "Balance before transaction",
    "newbalanceOrig": "Balance after transaction",
    "orig_amount_mean": "Customer normal amount",
    "orig_amount_std": "Customer spending variability",
    "time_since_prev": "Time since previous transaction",
    "tx_count_24h": "Transactions in last 24 hours",
    "merchant_risk": "Merchant risk",
    "device_account_count": "Accounts using this device",
    "ip_account_count": "Accounts using this network",
    "is_new_device": "New device",
    "is_new_ip": "New network address",
    "is_new_beneficiary": "New beneficiary",
    "is_new_merchant": "New merchant",
    "rapid_withdrawal_count": "Rapid withdrawal activity",
    "customer_category_deviation": "Spending category change",
    "location_available": "Location data available",
    "location_change_distance_km": "Distance from previous location",
    "impossible_travel_flag": "Impossible travel pattern",
}

PRESENTATION_MODELS = [
    ("Logistic Regression", "Interpretable linear baseline"),
    ("Decision Tree", "Rule-based non-linear model"),
    ("XGBoost", "Gradient-boosted tree model"),
    ("Random Forest", "Bagged tree ensemble"),
    ("Soft Voting", "Average of model probabilities"),
    ("Weighted Voting", "Performance-weighted ensemble"),
    ("Stacking", "Meta-model over base predictions"),
    ("Adaptive Ensemble", "Dynamic model selection"),
    ("SelectKBest Feature Selection", "Classical feature-selection baseline"),
    ("Quantum Genetic Feature Selection", "Quantum-inspired feature search"),
    ("Quantum Differential Evolution", "Quantum-inspired differential search"),
]

def build_named_transaction(row):
    """Build the current PaySim feature contract from a form or CSV row."""
    def number(name, default=0.0):
        value = row.get(name, default)
        return default if pd.isna(value) else float(value)

    transaction_type = str(row.get("type", row.get("transaction_type", "PAYMENT"))).upper()
    merchant = str(row.get("merchant", row.get("description", ""))).lower()
    channel = (
        "ATM" if transaction_type in {"CASH_OUT", "WITHDRAW"} else
        "BANK_TRANSFER" if transaction_type in {"TRANSFER", "CASH_IN", "DEPOSIT"} else
        "POS" if transaction_type in {"DEBIT", "POS"} else "ONLINE"
    )
    payload = {name: 0.0 for name in MODEL_FEATURE_NAMES}
    payload.update({
        "step": number("step", number("Time")),
        "time_hour": number("time_hour", number("Time") % 24),
        "amount": abs(number("amount", number("Amount"))),
        "oldbalanceOrg": number("oldbalanceOrg", number("balance_before")),
        "newbalanceOrig": number("newbalanceOrig", number("balance_after")),
        "oldbalanceDest": number("oldbalanceDest"),
        "newbalanceDest": number("newbalanceDest"),
        "orig_amount_mean": number("orig_amount_mean"),
        "orig_amount_std": number("orig_amount_std"),
        "time_since_prev": number("time_since_prev", -1),
        "tx_count_24h": number("tx_count_24h", 1),
        "amount_z": number("amount_z"),
        "is_unusual_amount": number("is_unusual_amount"),
        "merchant_risk": number("merchant_risk", number("merchantRisk", 0.4)),
        "account_age_days": number("account_age_days", 365),
        "auth_failures": number("auth_failures"),
        "device_account_count": number("device_account_count", 1),
        "ip_account_count": number("ip_account_count", 1),
        "beneficiary_tx_count": number("beneficiary_tx_count", 1),
        "merchant_tx_count": number("merchant_tx_count", 1),
        "is_new_device": number("is_new_device", 1),
        "is_new_ip": number("is_new_ip", 1),
        "is_new_beneficiary": number("is_new_beneficiary", 1),
        "is_new_merchant": number("is_new_merchant", 1),
        "rapid_withdrawal_count": number("rapid_withdrawal_count"),
        "customer_category_deviation": number("customer_category_deviation"),
        "type_CASH_OUT": 1.0 if transaction_type in {"CASH_OUT", "WITHDRAW"} else 0.0,
        "type_DEBIT": 1.0 if transaction_type == "DEBIT" else 0.0,
        "type_PAYMENT": 1.0 if transaction_type in {"PAYMENT", "POS"} else 0.0,
        "type_TRANSFER": 1.0 if transaction_type in {"TRANSFER", "WIRE"} else 0.0,
        "merchant_cat_grocery": 1.0 if any(word in merchant for word in ("grocery", "market", "food")) else 0.0,
        "merchant_cat_restaurant": 1.0 if any(word in merchant for word in ("restaurant", "cafe", "food")) else 0.0,
        "merchant_cat_retail": 1.0 if any(word in merchant for word in ("retail", "shop", "store", "mart")) else 0.0,
        "merchant_cat_travel": 1.0 if any(word in merchant for word in ("travel", "flight", "hotel")) else 0.0,
        "merchant_cat_utilities": 1.0 if any(word in merchant for word in ("utility", "telecom", "bill")) else 0.0,
        f"transaction_channel_{channel}": 1.0 if f"transaction_channel_{channel}" in MODEL_FEATURE_NAMES else 0.0,
    })
    return payload

def explain_named_result(result):
    explanation = result.get("explanation") or {}
    impacts = explanation.get("feature_impacts") or []
    table = []
    for impact in impacts:
        feature = impact.get("feature", "feature")
        item = dict(impact)
        item["feature"] = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
        table.append(item)
    return explanation, pd.DataFrame(table)

def main():
    st.title("Credit Card Fraud Detection System")
    st.markdown("""
    A three-layer fraud detection system with quantum-inspired optimization for real-time transaction screening.
    """)

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Choose a page:", [
        "Single Transaction Check",
        "Batch Analysis",
        "Model Performance",
        "System Architecture"
    ])

    # Check API connection
    api_status = check_api_status()
    if api_status:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Not Connected")
        st.error("Please start the API server by running: `uvicorn api:app --reload`")

    if page == "Single Transaction Check":
        single_transaction_page()
    elif page == "Batch Analysis":
        batch_analysis_page()
    elif page == "Model Performance":
        model_performance_page()
    elif page == "System Architecture":
        architecture_page()

def check_api_status():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def single_transaction_page():
    """Single transaction fraud detection page"""
    st.header("Single Transaction Fraud Detection")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Transaction Details")

        with st.form("transaction_form"):
            col_a, col_b = st.columns(2)

            with col_a:
                step = st.number_input("Step", min_value=0.0, value=1.0, step=1.0)
                transaction_type = st.selectbox("Transaction type", ["PAYMENT", "CASH_OUT", "TRANSFER", "DEBIT", "CASH_IN"])
                amount = st.number_input("Amount", min_value=0.0, value=0.0, step=0.01)
                balance_before = st.number_input("Balance before transaction", min_value=0.0, value=0.0, step=0.01)
                balance_after = st.number_input("Balance after transaction", min_value=0.0, value=0.0, step=0.01)
                merchant = st.text_input("Merchant or description", value="")
                merchant_risk = st.slider("Merchant risk", min_value=0.0, max_value=1.0, value=0.4, step=0.01)

            with col_b:
                time_hour = st.number_input("Transaction hour", min_value=0, max_value=23, value=12, step=1)
                tx_count_24h = st.number_input("Transactions in last 24 hours", min_value=0, value=1, step=1)
                time_since_prev = st.number_input("Time since previous transaction", min_value=-1.0, value=1.0, step=1.0)
                orig_amount_mean = st.number_input("Customer normal amount", min_value=0.0, value=0.0, step=0.01)
                orig_amount_std = st.number_input("Customer spending variability", min_value=0.0, value=0.0, step=0.01)
                device_account_count = st.number_input("Accounts using this device", min_value=1, value=1, step=1)
                ip_account_count = st.number_input("Accounts using this network", min_value=1, value=1, step=1)
                is_new_device = st.checkbox("New device")
                is_new_beneficiary = st.checkbox("New beneficiary")
                rapid_withdrawal_count = st.number_input("Rapid withdrawals", min_value=0, value=0, step=1)

            submitted = st.form_submit_button("🔍 Check for Fraud")

    with col2:
        st.subheader("Results")

        if submitted and check_api_status():
            transaction_data = build_named_transaction({
                "step": step, "type": transaction_type, "amount": amount,
                "oldbalanceOrg": balance_before, "newbalanceOrig": balance_after,
                "merchant": merchant, "merchant_risk": merchant_risk,
                "time_hour": time_hour, "tx_count_24h": tx_count_24h,
                "time_since_prev": time_since_prev, "orig_amount_mean": orig_amount_mean,
                "orig_amount_std": orig_amount_std, "device_account_count": device_account_count,
                "ip_account_count": ip_account_count, "is_new_device": int(is_new_device),
                "is_new_beneficiary": int(is_new_beneficiary),
                "rapid_withdrawal_count": rapid_withdrawal_count,
            })

            try:
                response = requests.post(f"{API_URL}/predict_named", json=transaction_data, timeout=10)
                response.raise_for_status()
                result = response.json()

                # Display results
                fraud_prob = result['fraud_probability']
                is_fraud = result['is_fraudulent']
                risk_level = result['risk_level']

                # Risk level color
                if risk_level == "High Risk":
                    color = "🔴"
                elif risk_level == "Medium Risk":
                    color = "🟠"
                elif risk_level == "Low Risk":
                    color = "🟡"
                else:
                    color = "🟢"

                st.metric("Fraud Probability", f"{fraud_prob:.1%}")
                st.metric("Risk Level", f"{color} {risk_level}")

                if is_fraud:
                    st.error("🚨 **FRAUDULENT TRANSACTION DETECTED!**")
                else:
                    st.success("✅ **Transaction appears legitimate**")

                # Model scores
                if 'model_scores' in result:
                    st.subheader("Model Scores")
                    scores_df = pd.DataFrame.from_dict(result['model_scores'], orient='index', columns=['Score'])
                    st.bar_chart(scores_df)

                # Quantum-enhanced score
                if result.get('quantum_enhanced_score') is not None:
                    st.metric("Quantum-Enhanced Score", f"{result['quantum_enhanced_score']:.3f}")

                if result.get('explanation'):
                    st.subheader("Why this transaction was flagged")
                    explanation, explanation_df = explain_named_result(result)
                    if explanation.get('reason'):
                        st.write(explanation['reason'])
                    if not explanation_df.empty:
                        st.dataframe(explanation_df[["feature", "value", "contribution", "direction"]])

            except Exception as e:
                st.error(f"Error processing transaction: {str(e)}")

        elif submitted:
            st.error("API not connected. Please start the server.")

def batch_analysis_page():
    """Batch analysis page"""
    st.header("📊 Batch Transaction Analysis")

    st.write("Upload a CSV file with multiple transactions for batch fraud detection.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())

            if st.button("🔍 Analyze Batch"):
                if check_api_status():
                    with st.spinner("Analyzing transactions..."):
                        # Process in batches to avoid timeout
                        batch_size = 50
                        results = []

                        for i in range(0, len(df), batch_size):
                            batch = df.iloc[i:i+batch_size]

                            for _, row in batch.iterrows():
                                transaction = build_named_transaction(row)
                                response = requests.post(f"{API_URL}/predict_named", json=transaction, timeout=30)
                                response.raise_for_status()
                                prediction = response.json()
                                results.append({
                                    "fraud_probability": prediction.get("fraud_probability", 0),
                                    "is_fraudulent": prediction.get("is_fraudulent", False),
                                    "risk_level": prediction.get("risk_level", ""),
                                    "model_scores": prediction.get("model_scores", {}),
                                    "explanation": prediction.get("explanation", {}),
                                    "source_row": len(results) + 1,
                                })

                        # Display results
                        results_df = pd.DataFrame(results)
                        st.write("Analysis Results:")
                        st.dataframe(results_df)

                        # Summary statistics
                        fraud_count = sum(1 for r in results if r['is_fraudulent'])
                        total_count = len(results)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Transactions", total_count)
                        with col2:
                            st.metric("Fraudulent", fraud_count)
                        with col3:
                            st.metric("Legitimate", total_count - fraud_count)

                        # Fraud probability distribution
                        probs = [r['fraud_probability'] for r in results]
                        fig, ax = plt.subplots()
                        ax.hist(probs, bins=20, alpha=0.7)
                        ax.set_xlabel('Fraud Probability')
                        ax.set_ylabel('Count')
                        ax.set_title('Fraud Probability Distribution')
                        st.pyplot(fig)

                else:
                    st.error("API not connected.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def model_performance_page():
    """Model performance visualization page"""
    st.header("📈 Model Performance Dashboard")

    if check_api_status():
        try:
            # Get model info
            response = requests.get(f"{API_URL}/model-info")
            model_info = response.json()

            st.subheader("Available Models")
            st.write(f"**Total Models:** {len(PRESENTATION_MODELS)}")

            cols = st.columns(3)
            for i, (model, purpose) in enumerate(PRESENTATION_MODELS):
                with cols[i % 3]:
                    st.code(model)

            # Mock performance data (in real implementation, you'd get this from API)
            st.subheader("Model Performance Metrics")

            # Sample performance data
            performance_data = {
                'Model': ['Logistic Regression', 'Decision Tree', 'XGBoost', 'Random Forest',
                         'Soft Voting', 'Weighted Voting', 'Stacking', 'Adaptive Ensemble'],
                'AUC': [0.95, 0.89, 0.97, 0.96, 0.98, 0.98, 0.97, 0.96],
                'Accuracy': [0.92, 0.85, 0.94, 0.93, 0.95, 0.95, 0.94, 0.93],
                'Precision': [0.88, 0.78, 0.91, 0.90, 0.93, 0.93, 0.92, 0.91],
                'Recall': [0.85, 0.82, 0.87, 0.86, 0.89, 0.90, 0.88, 0.87]
            }

            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df)

            # Performance visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(performance_data['Model']))
            width = 0.2

            ax.bar(x - width*1.5, performance_data['AUC'], width, label='AUC', alpha=0.8)
            ax.bar(x - width/2, performance_data['Accuracy'], width, label='Accuracy', alpha=0.8)
            ax.bar(x + width/2, performance_data['Precision'], width, label='Precision', alpha=0.8)
            ax.bar(x + width*1.5, performance_data['Recall'], width, label='Recall', alpha=0.8)

            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels([m.split()[0] for m in performance_data['Model']], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error loading model information: {str(e)}")
    else:
        st.error("API not connected.")

def architecture_page():
    """System architecture explanation page"""
    st.header("🏗️ System Architecture")

    st.markdown("""
    ## Three-Layer Fraud Detection System

    ### Layer 1: Real-Time Transaction Screening
    **Lightweight Models for Fast Processing:**
    - **Logistic Regression**: Simple, interpretable baseline model
    - **Decision Tree**: Captures non-linear relationships
    - **XGBoost**: Gradient boosting for high performance
    - **Random Forest**: Ensemble of decision trees for robustness

    ### Layer 2: Ensemble Intelligence
    **Combining Models for Better Performance:**
    - **Soft Voting**: Average probability predictions
    - **Weighted Voting**: Weighted combination based on model performance
    - **Stacking**: Meta-learner combines base model predictions
    - **Adaptive Ensemble**: Dynamically selects best models

    ### Layer 3: Quantum-Inspired Optimization
    **Advanced Techniques for Superior Performance:**
    - **Feature Selection**: Quantum-inspired algorithm selects optimal features
    - **Hyperparameter Optimization**: Inspired by quantum search algorithms
    - **Anomaly Scoring**: Quantum-enhanced outlier detection
    - **Risk Ranking**: Advanced risk assessment using quantum principles
    """)

    # Architecture diagram
    st.subheader("System Flow")
    st.graphviz_chart("""
    digraph {
        rankdir=TB;
        node [shape=box, style=filled, fillcolor=lightblue];
        
        A [label="Transaction Input"];
        B [label="Layer 1: Base Models\n(LR, DT, XGB, RF)"];
        C [label="Layer 2: Ensembles\n(Soft, Weighted, Stacking, Adaptive)"];
        D [label="Layer 3: Quantum-Inspired\n(Feature Selection, Optimization)"];
        E [label="Fraud Score & Risk Assessment"];
        
        A -> B -> C -> D -> E;
        
        subgraph cluster_legend {
            label="Legend";
            style=filled;
            color=lightgrey;
            L1 [label="Real-time Processing", shape=plaintext];
            L2 [label="Ensemble Methods", shape=plaintext];
            L3 [label="Advanced Optimization", shape=plaintext];
        }
    }
    """)

    st.markdown("""
    ## Key Features

    - **Real-time Processing**: Sub-millisecond prediction times
    - **High Accuracy**: Ensemble methods reduce false positives/negatives
    - **Scalable**: Handles high transaction volumes
    - **Interpretable**: Model explanations for regulatory compliance
    - **Adaptive**: Continuously learns and improves
    """)

if __name__ == "__main__":
    main()