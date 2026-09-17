"""
FastAPI Backend for Credit Card Fraud Detection System
"""

from fastapi import FastAPI, HTTPException  
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 
import numpy as np
import joblib
import os
from app import FraudDetectionSystem
import pandas as pd

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="""A three-layer fraud detection system with quantum-inspired optimization""",
    version="1.0.0"
)

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
fraud_system = None

class TransactionFeatures(BaseModel):
    """Transaction features for fraud detection"""
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    fraud_probability: float
    is_fraudulent: bool
    risk_level: str
    model_scores: dict
    quantum_enhanced_score: float = None
    explanation: dict = None

class SimpleTransaction(BaseModel):
    """Simplified transaction representation for app usage"""
    amount: float
    transaction_type: str = 'payment'
    timestamp: float = 0.0
    description: str = ''
    recipient: str = ''

@app.post("/predict_simple", response_model=PredictionResponse)
async def predict_simple(transaction: SimpleTransaction):
    """Predict fraud for a simplified transaction payload"""
    if not fraud_system:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Build a synthetic full credit card feature vector
        # Use amount and time while leaving PCA components as neutral values.
        features = np.zeros((1, 30), dtype=float)
        features[0, 0] = float(transaction.timestamp)
        features[0, -1] = float(transaction.amount)

        scores = fraud_system.get_real_time_score(features)
        explanation = scores.pop('explanation', None)
        fraud_probability = scores.get('soft_voting', 0.5)
        is_fraudulent = fraud_probability > 0.5

        if fraud_probability > 0.8:
            risk_level = "High Risk"
        elif fraud_probability > 0.6:
            risk_level = "Medium Risk"
        elif fraud_probability > 0.4:
            risk_level = "Low Risk"
        else:
            risk_level = "Very Low Risk"

        return PredictionResponse(
            fraud_probability=fraud_probability,
            is_fraudulent=is_fraudulent,
            risk_level=risk_level,
            model_scores=scores,
            quantum_enhanced_score=scores.get('quantum_enhanced'),
            explanation=explanation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load the trained model on startup"""
    global fraud_system
    try:
        # Prefer loading artifacts produced by training script
        fraud_system = FraudDetectionSystem()
        loaded = fraud_system.load_from_artifacts(model_dir='models')
        if loaded:
            print('Loaded model artifacts from models/')
            return

        # Fallback: try to load legacy saved object
        if os.path.exists('fraud_detection_model.pkl'):
            print("Loading saved model...")
            fraud_system = joblib.load('fraud_detection_model.pkl')
            print("Model loaded successfully!")
            return

        # Last resort: train from creditcard.csv if available
        if os.path.exists('creditcard.csv'):
            print("No artifacts found; training new model from creditcard.csv...")
            df = pd.read_csv('creditcard.csv')
            expected_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0.0
            X = df[expected_cols]
            y = df['Class']

            fraud_system.fit(X, y)
            # save legacy object for quick reuse
            joblib.dump(fraud_system, 'fraud_detection_model.pkl')
            print("Model trained and saved (legacy object)!")
            return

        print('No model artifacts or datasets available on startup; API will be inactive until model is provided.')

    except Exception as e:
        print(f"Error loading/training model: {e}")
        fraud_system = None


@app.post("/predict_named")
async def predict_named(features: dict):
    """Predict using a dictionary of named features mapped to model input."""
    if not fraud_system:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = fraud_system.prepare_named_features(features)
        if 'rf_paysim' in fraud_system.models:
            model_name = 'rf_paysim'
        else:
            model_name = next(iter(fraud_system.models.keys()))

        probs = fraud_system.predict(X, model_name=model_name)
        prob = float(probs[0])

        if prob > 0.8:
            risk_level = 'High Risk'
        elif prob > 0.6:
            risk_level = 'Medium Risk'
        elif prob > 0.4:
            risk_level = 'Low Risk'
        else:
            risk_level = 'Very Low Risk'

        explanation = fraud_system.explain_transaction(X, model_name=model_name)
        return {
            'fraud_probability': prob,
            'is_fraudulent': bool(prob > 0.5),
            'risk_level': risk_level,
            'model_scores': {model_name: prob},
            'explanation': explanation,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/explain_named")
async def explain_named(features: dict):
    """Explain prediction for a dictionary of named features."""
    if not fraud_system:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = fraud_system.prepare_named_features(features)
        model_name = next(iter(fraud_system.models.keys()))
        explanation = fraud_system.explain_transaction(X, model_name=model_name)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Card Fraud Detection API",
        "status": "active" if fraud_system else "inactive",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if fraud_system else "unhealthy",
        "model_loaded": fraud_system is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionFeatures):
    """Predict fraud probability for a transaction"""
    if not fraud_system:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert transaction to feature array
        features = np.array([[
            transaction.Time, transaction.V1, transaction.V2, transaction.V3,
            transaction.V4, transaction.V5, transaction.V6, transaction.V7,
            transaction.V8, transaction.V9, transaction.V10, transaction.V11,
            transaction.V12, transaction.V13, transaction.V14, transaction.V15,
            transaction.V16, transaction.V17, transaction.V18, transaction.V19,
            transaction.V20, transaction.V21, transaction.V22, transaction.V23,
            transaction.V24, transaction.V25, transaction.V26, transaction.V27,
            transaction.V28, transaction.Amount
        ]])

        # Get real-time scores
        scores = fraud_system.get_real_time_score(features)
        explanation = scores.pop('explanation', None)

        # Main fraud probability (from soft voting ensemble)
        fraud_probability = scores.get('soft_voting', 0.5)

        # Determine if fraudulent
        is_fraudulent = fraud_probability > 0.5

        # Risk level
        if fraud_probability > 0.8:
            risk_level = "High Risk"
        elif fraud_probability > 0.6:
            risk_level = "Medium Risk"
        elif fraud_probability > 0.4:
            risk_level = "Low Risk"
        else:
            risk_level = "Very Low Risk"

        # Quantum-enhanced score
        quantum_score = scores.get('quantum_enhanced')

        return PredictionResponse(
            fraud_probability=fraud_probability,
            is_fraudulent=is_fraudulent,
            risk_level=risk_level,
            model_scores=scores,
            quantum_enhanced_score=quantum_score,
            explanation=explanation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded models"""
    if not fraud_system:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "available_models": list(fraud_system.models.keys()),
        "is_trained": fraud_system.is_trained,
        "features_selected": len(fraud_system.quantum_results.get('selected_features', [])) if hasattr(fraud_system, 'quantum_results') else 0
    }

@app.post("/batch-predict")
async def batch_predict(transactions: list[TransactionFeatures]):
    """Predict fraud for multiple transactions"""
    if not fraud_system:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = []
        for transaction in transactions:
            # Reuse the single prediction logic
            features = np.array([[
                transaction.Time, transaction.V1, transaction.V2, transaction.V3,
                transaction.V4, transaction.V5, transaction.V6, transaction.V7,
                transaction.V8, transaction.V9, transaction.V10, transaction.V11,
                transaction.V12, transaction.V13, transaction.V14, transaction.V15,
                transaction.V16, transaction.V17, transaction.V18, transaction.V19,
                transaction.V20, transaction.V21, transaction.V22, transaction.V23,
                transaction.V24, transaction.V25, transaction.V26, transaction.V27,
                transaction.V28, transaction.Amount
            ]])

            scores = fraud_system.get_real_time_score(features)
            fraud_probability = scores.get('soft_voting', 0.5)
            is_fraudulent = fraud_probability > 0.5

            results.append({
                "fraud_probability": fraud_probability,
                "is_fraudulent": is_fraudulent,
                "scores": scores
            })

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/explain")
async def explain_transaction(transaction: TransactionFeatures):
    """Return a human-readable explanation for a transaction prediction"""
    if not fraud_system:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = np.array([[
            transaction.Time, transaction.V1, transaction.V2, transaction.V3,
            transaction.V4, transaction.V5, transaction.V6, transaction.V7,
            transaction.V8, transaction.V9, transaction.V10, transaction.V11,
            transaction.V12, transaction.V13, transaction.V14, transaction.V15,
            transaction.V16, transaction.V17, transaction.V18, transaction.V19,
            transaction.V20, transaction.V21, transaction.V22, transaction.V23,
            transaction.V24, transaction.V25, transaction.V26, transaction.V27,
            transaction.V28, transaction.Amount
        ]])

        scores = fraud_system.get_real_time_score(features)
        explanation = scores.get('explanation')
        model_scores = {k: v for k, v in scores.items() if k != 'explanation'}

        return {
            'explanation': explanation,
            'model_scores': model_scores
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)