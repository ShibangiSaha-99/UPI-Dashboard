import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb

# --- Configuration --- #
FILE_PATH = 'Upi fraud dataset final.csv'
XGB_OPTIMAL_THRESHOLD = 0.25
RISK_MEDIUM_THRESHOLD_PERCENT = 22
RISK_HIGH_THRESHOLD_PERCENT = 75

# --- Load pre-trained model components --- #
@st.cache_resource
def load_model_components():
    try:
        model = joblib.load("xgb_model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return None, None, None

# --- Data Loading and Preprocessing --- #
def load_and_preprocess_data(scaler, feature_columns):
    try:
        upi_df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        st.error(f"File not found: {FILE_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    # Drop high-uniqueness columns
    columns_to_drop = [col for col in upi_df.columns if (upi_df[col].nunique() / len(upi_df)) * 100 > 95]
    if columns_to_drop:
        upi_df.drop(columns=columns_to_drop, inplace=True)

    # Encode categorical columns
    categorical_cols = upi_df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        upi_df = pd.get_dummies(upi_df, columns=categorical_cols, drop_first=True)

    X_full = upi_df.drop('fraud', axis=1)
    y_full = upi_df['fraud']

    # Add missing feature columns if necessary
    for c in set(feature_columns) - set(X_full.columns):
        X_full[c] = 0

    X_full = X_full[feature_columns]

    # Scale features
    X_full_scaled = scaler.transform(X_full)
    X_full_processed = pd.DataFrame(X_full_scaled, columns=feature_columns, index=X_full.index)

    return X_full_processed, y_full

# --- Evaluate XGBoost Model --- #
def evaluate_xgb_model(_model, X_data, y_data, initial_optimal_threshold, current_threshold=None):
    y_pred_proba_xgb = _model.predict_proba(X_data)[:, 1]
    threshold_for_metrics = current_threshold if current_threshold is not None else initial_optimal_threshold
    y_pred_optimized_xgb = (y_pred_proba_xgb > threshold_for_metrics).astype(int)

    return {
        'y_pred_proba': y_pred_proba_xgb,
        'f1_optimized': f1_score(y_data, y_pred_optimized_xgb, pos_label=1),
        'recall_optimized': recall_score(y_data, y_pred_optimized_xgb, pos_label=1),
        'precision_optimized': precision_score(y_data, y_pred_optimized_xgb, pos_label=1),
        'roc_auc': roc_auc_score(y_data, y_pred_proba_xgb),
        'conf_matrix': confusion_matrix(y_data, y_pred_optimized_xgb),
        'roc_curve_data': roc_curve(y_data, y_pred_proba_xgb) + (roc_auc_score(y_data, y_pred_proba_xgb),)
    }

# --- Streamlit App --- #
def streamlit_app():
    st.title("UPI Fraud Detection Dashboard")
    st.write("A comprehensive dashboard for fraud detection using XGBoost model and a simple risk scoring system.")
    st.write("Current files in directory:")
    st.write(os.listdir())

    # Initialize session state
    for key in ['y_pred_proba_xgb_st', 'y_full_st', 'roc_curve_data_xgb_st', 'xgb_model_loaded']:
        if key not in st.session_state:
            st.session_state[key] = None if 'loaded' not in key else False

    # Load model components
    xgb_model_loaded, scaler_loaded, feature_columns_loaded = load_model_components()
    if None in [xgb_model_loaded, scaler_loaded, feature_columns_loaded]:
        st.stop()

    # --- Data Loading ---
    st.subheader("1. Data Loading and Preprocessing")
    st.info("Loading and preprocessing data... This may take a moment.")
    X_full_processed_st, y_full_st = load_and_preprocess_data(scaler_loaded, feature_columns_loaded)
    if X_full_processed_st is None or y_full_st is None:
        st.stop()
    st.success("Data loaded and preprocessed successfully!")
    st.write(f"Shape of processed X_full: {X_full_processed_st.shape}")

    # --- Sidebar for Hyperparameters ---
    st.sidebar.header("XGBoost Model Parameters (Pre-trained)")
    st.sidebar.info("These parameters reflect the pre-trained model's configuration.")
    st.sidebar.slider('n_estimators', 50, 200, xgb_model_loaded.n_estimators, 10, disabled=True)
    st.sidebar.slider('max_depth', 3, 10, xgb_model_loaded.max_depth, 1, disabled=True)
    st.sidebar.slider('learning_rate', 0.001, 0.5, float(xgb_model_loaded.learning_rate), 0.001, disabled=True)

    # --- Load and Evaluate Model ---
    if st.sidebar.button('Load and Evaluate Model'):
        st.subheader("2. XGBoost Model Evaluation")
        st.info("Evaluating XGBoost model performance...")
        xgb_eval_results = evaluate_xgb_model(xgb_model_loaded, X_full_processed_st, y_full_st, XGB_OPTIMAL_THRESHOLD)
        st.session_state['y_pred_proba_xgb_st'] = xgb_eval_results['y_pred_proba']
        st.session_state['y_full_st'] = y_full_st
        st.session_state['roc_curve_data_xgb_st'] = xgb_eval_results['roc_curve_data']
        st.session_state['xgb_model_loaded'] = True
        st.success("XGBoost Model evaluation complete!")

        # Display metrics
        st.write(f"**Optimal Threshold**: {XGB_OPTIMAL_THRESHOLD:.2f}")
        st.write(f"Precision: {xgb_eval_results['precision_optimized']:.4f}")
        st.write(f"Recall: {xgb_eval_results['recall_optimized']:.4f}")
        st.write(f"F1-Score: {xgb_eval_results['f1_optimized']:.4f}")
        st.write(f"ROC-AUC: {xgb_eval_results['roc_auc']:.4f}")
        st.write("**Confusion Matrix**")
        st.write(xgb_eval_results['conf_matrix'])

        # Plot ROC
        fpr_xgb, tpr_xgb, _, roc_auc_xgb = xgb_eval_results['roc_curve_data']
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr_xgb, tpr_xgb, label=f'ROC (area = {roc_auc_xgb:.2f})', color='blue')
        tn, fp, fn, tp = confusion_matrix(y_full_st, (xgb_eval_results['y_pred_proba'] > XGB_OPTIMAL_THRESHOLD).astype(int)).ravel()
        ax.plot(fp/(fp+tn), tp/(tp+fn), 'o', color='green', markersize=8, label=f'Optimal Thresh ({XGB_OPTIMAL_THRESHOLD:.2f})')
        ax.plot([0,1],[0,1],'--', color='red')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve for XGBoost')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # --- Risk Scoring ---
        st.header("3. Transaction Risk Scoring")
        risk_scores = xgb_eval_results['y_pred_proba'] * 100
        risk_categories = ['Low Risk' if s<RISK_MEDIUM_THRESHOLD_PERCENT else 'Medium Risk' if s<RISK_HIGH_THRESHOLD_PERCENT else 'High Risk' for s in risk_scores]
        risk_series = pd.Series(risk_categories, name='Risk Category')
        st.write("**Transaction counts:**")
        st.write(risk_series.value_counts())
        st.write("**Transaction percentages:**")
        st.write(risk_series.value_counts(normalize=True)*100)
        st.markdown("---")

    # --- Interactive Threshold Optimization ---
    st.header("4. Interactive Threshold Optimization")
    if st.session_state['xgb_model_loaded'] and st.session_state['y_pred_proba_xgb_st'] is not None:
        current_threshold_xgb = st.slider('Select Threshold', 0.0, 1.0, XGB_OPTIMAL_THRESHOLD, 0.01)
        y_pred_dynamic_xgb = (st.session_state['y_pred_proba_xgb_st'] > current_threshold_xgb).astype(int)
        precision_dynamic = precision_score(st.session_state['y_full_st'], y_pred_dynamic_xgb, pos_label=1)
        recall_dynamic = recall_score(st.session_state['y_full_st'], y_pred_dynamic_xgb, pos_label=1)
        f1_dynamic = f1_score(st.session_state['y_full_st'], y_pred_dynamic_xgb, pos_label=1)
        conf_matrix_dynamic = confusion_matrix(st.session_state['y_full_st'], y_pred_dynamic_xgb)

        st.write(f"**Metrics at Threshold {current_threshold_xgb:.2f}:**")
        st.write(f"Precision: {precision_dynamic:.4f}")
        st.write(f"Recall: {recall_dynamic:.4f}")
        st.write(f"F1-Score: {f1_dynamic:.4f}")
        st.write("**Confusion Matrix:**")
        st.write(conf_matrix_dynamic)

        # ROC plot
        fpr_xgb, tpr_xgb, _, roc_auc_xgb = st.session_state['roc_curve_data_xgb_st']
        fig_dyn, ax_dyn = plt.subplots(figsize=(8, 6))
        ax_dyn.plot(fpr_xgb, tpr_xgb, label=f'ROC (area={roc_auc_xgb:.2f})', color='blue')
        tn, fp, fn, tp = conf_matrix_dynamic.ravel()
        ax_dyn.plot(fp/(fp+tn), tp/(tp+fn), 'o', color='green', markersize=8, label=f'Current Thresh ({current_threshold_xgb:.2f})')
        ax_dyn.plot([0,1],[0,1],'--', color='red')
        ax_dyn.set_xlabel('False Positive Rate')
        ax_dyn.set_ylabel('True Positive Rate')
        ax_dyn.set_title('ROC Curve for XGBoost')
        ax_dyn.legend()
        ax_dyn.grid(True)
        st.pyplot(fig_dyn)
    else:
        st.warning("Load and evaluate the model first to enable threshold optimization.")

# --- Run the app --- #
if __name__ == '__main__':
    streamlit_app()
