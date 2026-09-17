import numpy as np
import joblib

fraud_system = joblib.load('fraud_detection_model.pkl')

transactions = [
    {'amount': 766.36, 'transaction_type': 'payment', 'timestamp': 17838.0, 'description': 'High-value online purchase', 'recipient': 'Amazon'},
    {'amount': 1.0, 'transaction_type': 'transfer', 'timestamp': 19762.0, 'description': 'Unknown micro transfer', 'recipient': 'External account'},
    {'amount': 1.0, 'transaction_type': 'payment', 'timestamp': 20011.0, 'description': 'Small suspicious charge', 'recipient': 'Merchant X'},
    {'amount': 1.0, 'transaction_type': 'withdrawal', 'timestamp': 8886.0, 'description': 'ATM cash withdrawal', 'recipient': 'ATM 421'},
    {'amount': 1.0, 'transaction_type': 'payment', 'timestamp': 8878.0, 'description': 'Recurring service charge', 'recipient': 'Subscription'},
    {'amount': 1.0, 'transaction_type': 'transfer', 'timestamp': 7535.0, 'description': 'Peer-to-peer transfer', 'recipient': 'Friend'},
    {'amount': 1.0, 'transaction_type': 'payment', 'timestamp': 14073.0, 'description': 'International purchase', 'recipient': 'Global Shop'},
    {'amount': 1.0, 'transaction_type': 'payment', 'timestamp': 14152.0, 'description': 'Suspicious transfer', 'recipient': 'Unknown merchant'},
    {'amount': 1.0, 'transaction_type': 'transfer', 'timestamp': 17520.0, 'description': 'External bank transfer', 'recipient': 'Other bank'},
    {'amount': 1.0, 'transaction_type': 'payment', 'timestamp': 8169.0, 'description': 'Unusual low-value charge', 'recipient': 'Retailer'}
]

for i, tx in enumerate(transactions, start=1):
    features = np.zeros((1, 30), dtype=float)
    features[0, 0] = float(tx['timestamp'])
    features[0, -1] = float(tx['amount'])
    scores = fraud_system.get_real_time_score(features)
    print(i, tx['amount'], tx['timestamp'], tx['transaction_type'], tx['description'], tx['recipient'], scores['soft_voting'], scores['weighted_voting'], scores['stacking'], scores['adaptive_ensemble'], scores['quantum_enhanced'])
