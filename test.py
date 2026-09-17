#!/usr/bin/env python3
"""
Test script for the Credit Card Fraud Detection System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import FraudDetectionSystem
import pandas as pd
import numpy as np

def test_data_loading():
    """Test data loading functionality"""
    print("Testing data loading...")
    try:
        df = pd.read_csv('creditcard.csv')
        print("Data loaded successfully: {} rows, {} columns".format(df.shape[0], df.shape[1]))
        print("Class distribution: {}".format(df['Class'].value_counts().to_dict()))
        return True
    except Exception as e:
        print("Error loading data: {}".format(e))
        return False

def test_model_initialization():
    """Test model initialization"""
    print("Testing model initialization...")
    try:
        fraud_system = FraudDetectionSystem()
        print("Fraud detection system initialized")
        return fraud_system
    except Exception as e:
        print("Error initializing system: {}".format(e))
        return None

def test_model_training(fraud_system):
    """Test model training"""
    print("Testing model training...")
    try:
        # Load a small sample for testing
        df = pd.read_csv('creditcard.csv')
        # Use a smaller sample for faster testing
        sample_size = min(10000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)

        X = df_sample.drop('Class', axis=1)
        y = df_sample['Class']

        X_test, y_test = fraud_system.fit(X, y)
        print("Model training completed")
        return X_test, y_test
    except Exception as e:
        print("Error training model: {}".format(e))
        return None, None

def test_predictions(fraud_system, X_test):
    """Test model predictions"""
    print("Testing predictions...")
    try:
        # Test single prediction
        sample_transaction = X_test[0]
        scores = fraud_system.get_real_time_score(sample_transaction)

        print("Single prediction successful")
        print("   Fraud probability: {:.4f}".format(scores.get('soft_voting', 'N/A')))

        # Test batch prediction
        batch_transactions = X_test[:5]
        batch_scores = []
        for transaction in batch_transactions:
            score = fraud_system.get_real_time_score(transaction)
            batch_scores.append(score)

        print("Batch prediction successful")
        return True
    except Exception as e:
        print("Error making predictions: {}".format(e))
        return False

def test_model_evaluation(fraud_system, X_test, y_test):
    """Test model evaluation"""
    print("Testing model evaluation...")
    try:
        results = fraud_system.evaluate_model(X_test, y_test, 'soft_voting')
        print("Model evaluation successful")
        print("   AUC Score: {:.4f}".format(results['auc_score']))
        return True
    except Exception as e:
        print("Error evaluating model: {}".format(e))
        return False

def run_all_tests():
    """Run all tests"""
    print("Running Credit Card Fraud Detection System Tests")
    print("=" * 60)

    # Test 1: Data Loading
    if not test_data_loading():
        return False

    # Test 2: Model Initialization
    fraud_system = test_model_initialization()
    if fraud_system is None:
        return False

    # Test 3: Model Training
    X_test, y_test = test_model_training(fraud_system)
    if X_test is None:
        return False

    # Test 4: Predictions
    if not test_predictions(fraud_system, X_test):
        return False

    # Test 5: Model Evaluation
    if not test_model_evaluation(fraud_system, X_test, y_test):
        return False

    print("\n" + "=" * 60)
    print("All tests passed! The system is working correctly.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_all_tests()
    exit_code = 0
    if not success:
        exit_code = 1
    sys.exit(exit_code)