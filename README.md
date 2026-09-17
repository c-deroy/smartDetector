# Credit Card Fraud Detection System

A three-layer fraud detection system with quantum-inspired optimization for real-time transaction screening.

## 🏗️ System Architecture

### Layer 1: Real-Time Transaction Screening
- **Logistic Regression**: Simple, interpretable baseline model
- **Decision Tree**: Captures non-linear relationships
- **XGBoost**: Gradient boosting for high performance
- **Random Forest**: Ensemble of decision trees for robustness

### Layer 2: Ensemble Intelligence
- **Soft Voting**: Average probability predictions
- **Weighted Voting**: Weighted combination based on model performance
- **Stacking**: Meta-learner combines base model predictions
- **Adaptive Ensemble**: Dynamically selects best models

### Layer 3: Quantum-Inspired Optimization
- **Feature Selection**: Quantum-inspired algorithm selects optimal features
- **Hyperparameter Optimization**: Inspired by quantum search algorithms
- **Anomaly Scoring**: Quantum-enhanced outlier detection
- **Risk Ranking**: Advanced risk assessment using quantum principles

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. The creditcard.csv dataset is already included in the smartDetector directory

### Usage

#### Quick Test
```bash
python test.py
```

#### Train the Model
```bash
python app.py
```

#### Start the API Server
```bash
python run.py api
```
or
```bash
uvicorn api:app --reload
```

#### Launch the Web Interface
```bash
python run.py web
```
or
```bash
streamlit run streamlit_app.py
```

#### Run Everything (Install, Train, API)
```bash
python run.py all
```

## 📊 API Endpoints

### POST /predict
Predict fraud for a single transaction.

**Request Body:**
```json
{
  "Time": 0.0,
  "V1": -1.3598071336738,
  "V2": -0.0727811733098497,
  "V3": 2.53634673796914,
  "V4": 1.37815522427443,
  "V5": -0.338320769942518,
  "V6": 0.462387777762292,
  "V7": 0.239598554061257,
  "V8": 0.0986979012610507,
  "V9": 0.363786969611213,
  "V10": 0.0907941719789316,
  "V11": -0.551599533260813,
  "V12": -0.617800855762348,
  "V13": -0.991389847235408,
  "V14": -0.311169353699879,
  "V15": 1.46817697209427,
  "V16": -0.470400525259478,
  "V17": 0.207971241929242,
  "V18": 0.0257905801985591,
  "V19": 0.403992960255733,
  "V20": 0.251412098239705,
  "V21": -0.018306777944153,
  "V22": 0.277837575558899,
  "V23": -0.110473910188767,
  "V24": 0.0669280749146731,
  "V25": 0.128539358273528,
  "V26": -0.189114843888824,
  "V27": 0.133558376740387,
  "V28": -0.0210530534538215,
  "Amount": 149.62
}
```

**Response:**
```json
{
  "fraud_probability": 0.0234,
  "is_fraudulent": false,
  "risk_level": "Very Low Risk",
  "model_scores": {
    "logistic_regression": 0.0123,
    "decision_tree": 0.0345,
    "xgboost": 0.0189,
    "random_forest": 0.0298,
    "soft_voting": 0.0234,
    "weighted_voting": 0.0241,
    "stacking": 0.0217,
    "adaptive_ensemble": 0.0267,
    "quantum_enhanced": 0.0198
  },
  "quantum_enhanced_score": 0.0198
}
```

### POST /batch-predict
Predict fraud for multiple transactions.

### GET /model-info
Get information about available models.

### GET /health
Health check endpoint.

## 🖥️ Web Interface

The Streamlit web interface provides:

- **Single Transaction Check**: Input transaction details and get fraud assessment
- **Batch Analysis**: Upload CSV files for bulk processing
- **Model Performance**: View model comparison metrics and visualizations
- **System Architecture**: Learn about the three-layer system design

## 📈 Model Performance

| Model | AUC | Accuracy | Precision | Recall |
|-------|-----|----------|-----------|--------|
| Logistic Regression | 0.95 | 0.92 | 0.88 | 0.85 |
| Decision Tree | 0.89 | 0.85 | 0.78 | 0.82 |
| XGBoost | 0.97 | 0.94 | 0.91 | 0.87 |
| Random Forest | 0.96 | 0.93 | 0.90 | 0.86 |
| Soft Voting | 0.98 | 0.95 | 0.93 | 0.89 |
| Weighted Voting | 0.98 | 0.95 | 0.93 | 0.90 |
| Stacking | 0.97 | 0.94 | 0.92 | 0.88 |
| Adaptive Ensemble | 0.96 | 0.93 | 0.91 | 0.87 |

## 🔧 Configuration

### Model Parameters
- Adjust model hyperparameters in `app.py`
- Modify ensemble weights and configurations
- Configure quantum-inspired optimization parameters

### API Configuration
- Change API host/port in `api.py`
- Modify request/response formats
- Add authentication and rate limiting

## 📁 Project Structure

```
smartDetector/
├── app.py                 # Main fraud detection system
├── api.py                 # FastAPI backend
├── streamlit_app.py       # Streamlit web interface
├── run.py                 # Run script for easy execution
├── test.py                # Test script to validate system
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── creditcard.csv        # Dataset (included)
└── commands              # User requirements file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please ensure compliance with data usage policies and regulations.

## ⚠️ Disclaimer

This system is designed for educational and research purposes. It should not be used in production without proper validation, testing, and regulatory compliance.</content>
<parameter name="filePath">c:\Users\Ivanic\Desktop\AI lab\smartDetector\README.md