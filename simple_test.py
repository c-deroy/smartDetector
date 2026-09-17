#!/usr/bin/env python3
"""
Simple test for the Credit Card Fraud Detection System
"""

print("Testing Credit Card Fraud Detection System...")

try:
    # Test import
    from app import FraudDetectionSystem
    print("✓ Import successful")

    # Test initialization
    fraud_system = FraudDetectionSystem()
    print("✓ System initialization successful")

    # Test data loading
    import pandas as pd
    df = pd.read_csv('creditcard.csv')
    print("✓ Data loading successful: {} rows".format(len(df)))

    print("\n🎉 Basic system test passed!")
    print("The three-layer fraud detection system is ready!")

except Exception as e:
    print("❌ Error: {}".format(e))
    print("Please check the system setup.")