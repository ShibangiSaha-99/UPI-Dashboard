import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
import matplotlib.pyplot as plt
import os

# --- Configuration --- #
FILE_PATH = "Upi fraud dataset final.csv"
XGB_MODEL_PATH = "xgb_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"

XGB_OPTIMAL_THRESHOLD = 0.25
RISK_MEDIUM_THRESHOLD_PERCENT = 22
RISK_HIGH_THRESHOLD_PERCENT = 75

# --- Load Model Components with caching --- #
@st.cache_resource
def load_model_components():
    try:
        model = joblib.load(XGB_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return None, None, None

# --- Load and preprocess data --- #
def load_and_preprocess_data(scaler, feature_columns):
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        st.error(f"CSV file not found: {FILE_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    # Drop ID-like columns (high uniqueness >95%)
    columns_to_drop = [col for col in df.columns if df[col].nunique()/len(df) > 0.95]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Convert categorical to dummy variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Separate X and y
    if 'fraud' not in df.columns:
        st.error("Target column 'fraud' not found in dataset.")
        st.stop()
    X = df.drop('fraud', axis=1)
    y = df['fraud']

    # Add missing feature columns
    for c in set(feature_columns) - set(X.columns):
        X[c] = 0
    X = X[feature_columns]

    # Scale features
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        st.stop()
    X_processed = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
    return X_processed, y

# --- Evaluate XGBoost --- #
def evaluate_xgb_model(model, X_data, y_data, threshold=XGB_OPTIMAL_THRESHOLD):
    y_proba = model.predict_proba(X_data)[:,1]
    y_pred = (y_proba > threshold).astype(int)

    f1 = f1_score(y_data, y_pred, pos_label=1)
    recall = recall_score(y_data, y_pred, pos_label=1)
    precision = precision_score(y_data, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_data, y_proba)
    conf_matrix = confusion_matrix(y_data, y_pred)
    fpr, tpr, _ = roc_curve(y_data, y_proba)

    return {
        "y_proba": y_proba,
        "y_pred": y_pred,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "roc_auc": roc_auc,
        "conf_matrix": conf_matrix,
        "roc_curve_data": (fpr, tpr)
    }

# --- Streamlit App --- #
def streamlit_app():
    st.title("UPI Fraud Detection Dashboard")
    st.write("Dashboard for fraud detection using a pre-trained XGBoost model and risk scoring system.")

    # Show files in repo
    st.subheader("Current files in repo:")
    st.write(os.listdir())

    # Load model components
    xgb_model, scaler, feature_columns = load_model_components()
    if xgb_model is None or scaler is None or feature_columns is None:
        st.stop()

    # Load and preprocess data
    st.subheader("1. Data Loading & Preprocessing")
    st.info("Loading data...")
    X_processed, y = load_and_preprocess_data(scaler, feature_columns)
    st.success("Data loaded & preprocessed successfully!")
    st.write(f"Processed dataset shape: {X_processed.shape}")

    # Sidebar: show pre-trained model params
    st.sidebar.header("XGBoost Model Parameters (Pre-trained)")
    st.sidebar.info("Parameters from pre-trained model (display only).")
    st.sidebar.text(f"n_estimators: {xgb_model.n_estimators}")
    st.sidebar.text(f"max_depth: {xgb_model.max_depth}")
    st.sidebar.text(f"learning_rate: {xgb_model.learning_rate}")

    # --- Load & Evaluate Model ---
    if st.sidebar.button("Load & Evaluate Model"):
        st.subheader("2. XGBoost Model Evaluation")
        st.info("Evaluating model...")
        results = evaluate_xgb_model(xgb_model, X_processed, y)
        st.success("Evaluation complete!")

        st.write(f"Optimal threshold: {XGB_OPTIMAL_THRESHOLD}")
        st.write(f"Precision (Fraud): {results['precision']:.4f}")
        st.write(f"Recall (Fraud): {results['recall']:.4f}")
        st.write(f"F1-Score (Fraud): {results['f1']:.4f}")
        st.write(f"ROC-AUC: {results['roc_auc']:.4f}")

        st.write("Confusion Matrix:")
        st.write(results['conf_matrix'])

        # ROC Curve
        fpr, tpr = results['roc_curve_data']
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(fpr, tpr, label=f'ROC curve (AUC={results["roc_auc"]:.2f})', color='blue')
        ax.plot([0,1],[0,1], linestyle='--', color='red')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)

        # --- Risk Scoring ---
        st.subheader("3. Risk Scoring")
        st.info("Converting probabilities to 0-100 risk scores.")

        risk_scores = results['y_proba']*100
        risk_categories = [
            "Low Risk" if s<RISK_MEDIUM_THRESHOLD_PERCENT else 
            "Medium Risk" if s<RISK_HIGH_THRESHOLD_PERCENT else
            "High Risk" 
            for s in risk_scores
        ]
        risk_series = pd.Series(risk_categories, name="Risk Category")
        st.write("Transactions per risk category:")
        st.write(risk_series.value_counts())
        st.write("Percentage per category:")
        st.write(risk_series.value_counts(normalize=True)*100)

    # --- Interactive Threshold Slider ---
    st.header("4. Interactive Threshold Optimization")
    if 'results' in locals():
        current_thresh = st.slider("Adjust Threshold", 0.0, 1.0, XGB_OPTIMAL_THRESHOLD, 0.01)
        y_pred_dynamic = (results['y_proba'] > current_thresh).astype(int)
        st.write(f"Precision: {precision_score(y, y_pred_dynamic):.4f}")
        st.write(f"Recall: {recall_score(y, y_pred_dynamic):.4f}")
        st.write(f"F1-Score: {f1_score(y, y_pred_dynamic):.4f}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y, y_pred_dynamic))
    else:
        st.info("Click 'Load & Evaluate Model' first to enable threshold slider.")

if __name__ == "__main__":
    streamlit_app()
