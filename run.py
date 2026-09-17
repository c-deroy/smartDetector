#!/usr/bin/env python3
"""
Run script for the Credit Card Fraud Detection System
"""

import os
import sys
import subprocess

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    return True

def run_training():
    """Run the model training"""
    print("Training the fraud detection model...")
    try:
        subprocess.check_call([sys.executable, "app.py"])
        print("Model training completed!")
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        return False
    return True

def run_api():
    """Run the FastAPI server"""
    print("Starting API server...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    try:
        subprocess.check_call([sys.executable, "-m", "uvicorn", "api:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])
    except subprocess.CalledProcessError as e:
        print(f"Error starting API: {e}")
        return False
    return True

def run_streamlit():
    """Run the Streamlit web interface"""
    print("Starting Streamlit web interface...")
    print("Web interface will be available at: http://localhost:8501")
    try:
        subprocess.check_call([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except subprocess.CalledProcessError as e:
        print(f"Error starting Streamlit: {e}")
        return False
    return True

def main():
    """Main function"""
    print("Credit Card Fraud Detection System")
    print("=" * 50)

    # Check if dataset exists
    if not os.path.exists('creditcard.csv'):
        print("Error: creditcard.csv not found!")
        print("Please download the dataset from: https://www.kaggle.com/mlg-ulb/creditcardfraud")
        print("Place it in the smartDetector directory.")
        return

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python run.py [command]")
        print("Commands:")
        print("  install    - Install dependencies")
        print("  train      - Train the model")
        print("  api        - Run API server")
        print("  web        - Run web interface")
        print("  all        - Install, train, and run API")
        return

    command = sys.argv[1].lower()

    if command == "install":
        install_dependencies()
    elif command == "train":
        run_training()
    elif command == "api":
        run_api()
    elif command == "web":
        run_streamlit()
    elif command == "all":
        if install_dependencies():
            if run_training():
                run_api()
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()