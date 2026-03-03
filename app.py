import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb

# --- Configuration --- #
FILE_PATH = 'Upi fraud dataset final.csv'
XGB_OPTIMAL_THRESHOLD = 0.25
RISK_MEDIUM_THRESHOLD_PERCENT = 22
RISK_HIGH_THRESHOLD_PERCENT = 75

# --- Function: Load Model Components --- #
def load_model_components():
    st.write("🔹 Loading pre-trained model components...")
    required_files = ["xgb_model.pkl", "scaler.pkl", "feature_columns.pkl"]
    for f in required_files:
        if not os.path.exists(f):
            st.error(f"❌ Missing file: {f}. Upload to the repo root.")
            return None, None, None

    try:
        model = joblib.load("xgb_model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        st.success("✅ Model components loaded successfully!")
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return None, None, None

# --- Function: Load and Preprocess Data --- #
def load_and_preprocess_data(scaler, feature_columns):
    st.write("🔹 Loading dataset...")
    if not os.path.exists(FILE_PATH):
        st.error(f"❌ Dataset file not found: {FILE_PATH}")
        return None, None

    try:
        df = pd.read_csv(FILE_PATH)
        st.write(f"Dataset loaded successfully! Shape: {df.shape}")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None, None

    # Drop high-uniqueness columns (IDs)
    columns_to_drop = [c for c in df.columns if df[c].nunique()/len(df) > 0.95]
    if columns_to_drop:
        st.write(f"Dropping ID-like columns: {columns_to_drop}")
        df.drop(columns=columns_to_drop, inplace=True)

    # One-hot encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.write(f"One-hot encoding columns: {list(categorical_cols)}")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Split features and target
    if 'fraud' not in df.columns:
        st.error("❌ Column 'fraud' missing from dataset!")
        return None, None

    X = df.drop('fraud', axis=1)
    y = df['fraud']

    # Add missing columns if any
    missing_cols = set(feature_columns) - set(X.columns)
    for c in missing_cols:
        X[c] = 0
    X = X[feature_columns]

    # Scale
    try:
        X_scaled = scaler.transform(X)
        X_processed = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
        st.success(f"✅ Data preprocessing complete! Shape: {X_processed.shape}")
        return X_processed, y
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        return None, None

# --- Function: Evaluate XGBoost Model --- #
def evaluate_xgb_model(model, X_data, y_data, threshold):
    y_pred_proba = model.predict_proba(X_data)[:, 1]
    y_pred = (y_pred_proba > threshold).astype(int)

    precision = precision_score(y_data, y_pred, pos_label=1)
    recall = recall_score(y_data, y_pred, pos_label=1)
    f1 = f1_score(y_data, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_data, y_pred_proba)
    conf_matrix = confusion_matrix(y_data, y_pred)
    fpr, tpr, _ = roc_curve(y_data, y_pred_proba)

    return {
        'y_pred_proba': y_pred_proba,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'conf_matrix': conf_matrix,
        'roc_curve': (fpr, tpr)
    }

# --- Streamlit App --- #
def streamlit_app():
    st.title("UPI Fraud Detection Dashboard")
    st.write("Interactive dashboard for UPI fraud detection using XGBoost.")

    st.subheader("Step 0: Check files in repo")
    st.write(os.listdir())

    # Load model components
    model, scaler, feature_columns = load_model_components()
    if model is None or scaler is None or feature_columns is None:
        st.stop()

    # Load and preprocess data
    X, y = load_and_preprocess_data(scaler, feature_columns)
    if X is None or y is None:
        st.stop()

    # Sidebar: model info
    st.sidebar.header("XGBoost Pre-trained Model Info")
    st.sidebar.write(f"n_estimators: {model.n_estimators}")
    st.sidebar.write(f"max_depth: {model.max_depth}")
    st.sidebar.write(f"learning_rate: {model.learning_rate}")

    # Evaluate model
    if st.sidebar.button("Evaluate Model"):
        st.subheader("XGBoost Model Evaluation")
        eval_results = evaluate_xgb_model(model, X, y, XGB_OPTIMAL_THRESHOLD)
        st.write(f"Precision: {eval_results['precision']:.4f}")
        st.write(f"Recall: {eval_results['recall']:.4f}")
        st.write(f"F1-score: {eval_results['f1']:.4f}")
        st.write(f"ROC-AUC: {eval_results['roc_auc']:.4f}")
        st.write("Confusion Matrix:")
        st.write(eval_results['conf_matrix'])

        # ROC Curve
        fpr, tpr = eval_results['roc_curve']
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {eval_results['roc_auc']:.2f})")
        ax.plot([0, 1], [0, 1], 'r--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)

        # Risk Scoring
        st.subheader("Transaction Risk Scoring")
        risk_scores = eval_results['y_pred_proba'] * 100
        risk_categories = ["Low Risk" if s < RISK_MEDIUM_THRESHOLD_PERCENT else
                           "Medium Risk" if s < RISK_HIGH_THRESHOLD_PERCENT else "High Risk"
                           for s in risk_scores]
        st.write(pd.Series(risk_categories).value_counts())
        st.write(pd.Series(risk_categories).value_counts(normalize=True)*100)

# --- Run App --- #
if __name__ == "__main__":
    streamlit_app()
