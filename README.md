# 🩺 Breast Cancer Detection AI Assistant
A Machine Learning web app deployed with Streamlit using XGBoost to predict breast cancer malignancy based on cell nuclei measurements. Demonstrates an end-to-end MLOps pipeline, including strict data validation, automated test cases, and probability calibration.

🏗️ Pipeline & Technical Architecture
1. Research & Model Training (01_entrenamiento_xgboost.ipynb)

Algorithm: Optimized XGBoost classifier chosen for its superior performance on tabular medical data and efficient gradient boosting.

Optimization: Focused on minimizing False Negatives (critical in healthcare) with high precision/recall calibration.

Serialization: Model exported via joblib for high-performance production inference.

2. Production Application (app2.py)

Engine: Built with Streamlit to provide a real-time clinical decision-support interface.

Data Integrity: Implements strict validation to block biological impossibilities and out-of-distribution noise (GIGO prevention).

Testing: Integrated automated clinical test cases (Malignant/Benign) for instant model verification.

🛠️ Tech Stack
Core: Python 3.11, XGBoost, Scikit-Learn.

Data: Pandas, NumPy, Joblib.

Deployment: Streamlit Cloud, GitHub.

🚀 Future Roadmap
Computer Vision: Transition from tabular data to raw image processing using ResNet architectures via PyTorch.

MLOps: Integration of GitHub Actions for CI/CD and Docker containerization.

Explainable AI (XAI): Implementation of SHAP values to provide visual evidence of feature importance in clinical predictions.

🔴 Live Demo: 

Disclaimer: This project is for educational and portfolio purposes only. It is not a medical device and should not be used for actual diagnosis.
