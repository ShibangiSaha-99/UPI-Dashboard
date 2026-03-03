import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb

# ------------------ Configuration ------------------ #
FILE_PATH = 'Upi fraud dataset final.csv'
XGB_OPTIMAL_THRESHOLD = 0.25
RISK_MEDIUM_THRESHOLD_PERCENT = 22
RISK_HIGH_THRESHOLD_PERCENT = 75

# ------------------ Functions ------------------ #
def load_model_components():
    required_files = ["xgb_model.pkl", "scaler.pkl", "feature_columns.pkl"]
    for f in required_files:
        if not os.path.exists(f):
            st.error(f"Missing file: {f}")
            return None, None, None
    try:
        model = joblib.load("xgb_model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return None, None, None

def load_and_preprocess_data(scaler, feature_columns):
    if not os.path.exists(FILE_PATH):
        st.error(f"Dataset file not found: {FILE_PATH}")
        return None, None
    df = pd.read_csv(FILE_PATH)

    # Drop ID-like columns
    columns_to_drop = [c for c in df.columns if df[c].nunique()/len(df) > 0.95]
    df.drop(columns=columns_to_drop, inplace=True)

    # One-hot encode categorical
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop('fraud', axis=1)
    y = df['fraud']

    missing_cols = set(feature_columns) - set(X.columns)
    for c in missing_cols:
        X[c] = 0
    X = X[feature_columns]

    X_scaled = scaler.transform(X)
    X_processed = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
    return X_processed, y

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

# ------------------ Streamlit App ------------------ #
st.title("UPI Fraud Detection Dashboard")
st.write("Files in repo:", os.listdir())

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

# ------------------ Interactive Threshold Slider ------------------ #
st.subheader("Interactive Threshold Optimization")
if 'y_pred_proba' not in st.session_state:
    st.session_state['y_pred_proba'] = model.predict_proba(X)[:,1]
    st.session_state['y_true'] = y

threshold_slider = st.slider("Select Classification Threshold", min_value=0.0, max_value=1.0,
                             value=XGB_OPTIMAL_THRESHOLD, step=0.01)

# Recompute metrics at slider threshold
y_pred_dynamic = (st.session_state['y_pred_proba'] > threshold_slider).astype(int)
precision_dyn = precision_score(st.session_state['y_true'], y_pred_dynamic, pos_label=1)
recall_dyn = recall_score(st.session_state['y_true'], y_pred_dynamic, pos_label=1)
f1_dyn = f1_score(st.session_state['y_true'], y_pred_dynamic, pos_label=1)
conf_matrix_dyn = confusion_matrix(st.session_state['y_true'], y_pred_dynamic)

st.write(f"Metrics at Threshold {threshold_slider:.2f}:")
st.write(f"Precision: {precision_dyn:.4f}")
st.write(f"Recall: {recall_dyn:.4f}")
st.write(f"F1-score: {f1_dyn:.4f}")
st.write("Confusion Matrix:")
st.write(conf_matrix_dyn)

# ROC curve with current threshold point
fpr, tpr, _ = roc_curve(st.session_state['y_true'], st.session_state['y_pred_proba'])
fig_dyn, ax_dyn = plt.subplots(figsize=(8,6))
ax_dyn.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_score(st.session_state['y_true'], st.session_state['y_pred_proba']):.2f})")
tn, fp, fn, tp = confusion_matrix(st.session_state['y_true'], y_pred_dynamic).ravel()
fpr_thresh = fp / (fp + tn)
tpr_thresh = tp / (tp + fn)
ax_dyn.plot(fpr_thresh, tpr_thresh, 'go', markersize=8, label=f'Threshold {threshold_slider:.2f}')
ax_dyn.plot([0, 1], [0, 1], 'r--')
ax_dyn.set_xlabel("False Positive Rate")
ax_dyn.set_ylabel("True Positive Rate")
ax_dyn.set_title("ROC Curve with Current Threshold")
ax_dyn.legend()
st.pyplot(fig_dyn)

# ------------------ Risk Scoring ------------------ #
risk_scores = st.session_state['y_pred_proba'] * 100
risk_categories = ["Low Risk" if s < RISK_MEDIUM_THRESHOLD_PERCENT else
                   "Medium Risk" if s < RISK_HIGH_THRESHOLD_PERCENT else "High Risk"
                   for s in risk_scores]

st.subheader("Transaction Risk Scoring")
st.write(pd.Series(risk_categories).value_counts())
st.write(pd.Series(risk_categories).value_counts(normalize=True)*100)

# ------------------ Download Predictions + Risk Scores ------------------ #
st.subheader("Download Predictions + Risk Scores")
results_df = pd.DataFrame({
    "Transaction_Index": X.index,
    "Predicted_Probability": st.session_state['y_pred_proba'],
    "Risk_Score": risk_scores,
    "Risk_Category": risk_categories,
    "Predicted_Label": y_pred_dynamic
})
csv = results_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download CSV (Current Threshold)",
    data=csv,
    file_name='upi_fraud_predictions_dynamic.csv',
    mime='text/csv'
)
