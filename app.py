import streamlit as st
import pandas as pd
import numpy as np
import pickle # Needed for loading pre-trained models
# Removed TensorFlow imports as per new plan
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Input
# from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# --- Configuration --- #
FILE_PATH = 'Upi fraud dataset final.csv' # Assumed to be in the same directory as app.py
TEST_SIZE = 0.2 # This is kept for reproducibility of preprocessing, though not directly used for splitting in app
RANDOM_STATE = 42
XGB_OPTIMAL_THRESHOLD = 0.25 # From previous analysis
RISK_MEDIUM_THRESHOLD_PERCENT = 22 # Corresponds to XGB_OPTIMAL_THRESHOLD * 100 or custom
RISK_HIGH_THRESHOLD_PERCENT = 75 # Corresponds to 0.75 * 100 or custom

# --- Model Component Loading --- #
@st.cache_resource
def load_model_components():
    try:
        # Assuming model components are in the same directory as the app.py
        with open('xgb_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        return xgb_model, scaler, feature_columns
    except FileNotFoundError:
        return None, None # Return None values on error
    except EOFError:
        return None, None, None # Return None values on error
    except Exception as e:
        return None, None # Return None values on error

# --- Data Loading and Preprocessing (using loaded scaler and feature columns) --- #
@st.cache_data
def load_and_preprocess_data(scaler, feature_columns):
    # Load the original UPI dataset
    try:
        upi_df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
    st.error(f"File not found: {FILE_PATH}")
    st.stop() # Explicitly return None on error
    except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop() # Explicitly return None on error

    # Drop ID-like columns (uniqueness > 95%)
    columns_to_drop = []
    for col in upi_df.columns:
        unique_values_count = upi_df[col].nunique()
        total_rows = len(upi_df)
        uniqueness_percentage = (unique_values_count / total_rows) * 100
        if uniqueness_percentage > 95:
            columns_to_drop.append(col)
    if columns_to_drop:
        upi_df.drop(columns=columns_to_drop, inplace=True)

    # One-hot encode categorical variables
    categorical_cols = upi_df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        upi_df = pd.get_dummies(upi_df, columns=categorical_cols, drop_first=True)

    # Separate features (X) and target variable (y)
    X_full = upi_df.drop('fraud', axis=1)
    y_full = upi_df['fraud']

    # Align columns of X_full with the feature_columns used during training
    # Add missing columns with 0
    missing_cols = set(feature_columns) - set(X_full.columns)
    for c in missing_cols:
        X_full[c] = 0
    # Drop columns not in feature_columns and ensure order
    X_full = X_full[feature_columns] # Ensure the order of feature columns is the same

    # Scale features using the loaded StandardScaler
    X_full_scaled = scaler.transform(X_full)

    # Convert scaled arrays back to DataFrames, preserving column names
    X_full_processed = pd.DataFrame(X_full_scaled, columns=feature_columns, index=X_full.index)

    return X_full_processed, y_full

# --- XGBoost Model Evaluation --- #
@st.cache_data
def evaluate_xgb_model(_model, X_data, y_data, initial_optimal_threshold, current_threshold=None):
    y_pred_proba_xgb = _model.predict_proba(X_data)[:, 1]

    # Determine which threshold to use for 'optimized' metrics
    threshold_for_metrics = current_threshold if current_threshold is not None else initial_optimal_threshold

    y_pred_optimized_xgb = (y_pred_proba_xgb > threshold_for_metrics).astype(int)
    f1_optimized_xgb = f1_score(y_data, y_pred_optimized_xgb, pos_label=1)
    recall_optimized_xgb = recall_score(y_data, y_pred_optimized_xgb, pos_label=1)
    precision_optimized_xgb = precision_score(y_data, y_pred_optimized_xgb, pos_label=1)

    conf_matrix_xgb = confusion_matrix(y_data, y_pred_optimized_xgb)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_data, y_pred_proba_xgb)

    return {
        'y_pred_proba': y_pred_proba_xgb,
        'f1_optimized': f1_optimized_xgb,
        'recall_optimized': recall_optimized_xgb,
        'precision_optimized': precision_optimized_xgb,
        'roc_auc': roc_auc_score(y_data, y_pred_proba_xgb),
        'conf_matrix': conf_matrix_xgb,
        'roc_curve_data': (fpr_xgb, tpr_xgb, roc_auc_score(y_data, y_pred_proba_xgb))
    }

def streamlit_app():
    # --- Streamlit App Title --- #
    st.title("UPI Fraud Detection Dashboard")
    st.write("A comprehensive dashboard for fraud detection using XGBoost model, "
             "and a simple risk scoring system.")

    # Initialize session state variables if they don't exist
    if 'y_pred_proba_xgb_st' not in st.session_state:
        st.session_state['y_pred_proba_xgb_st'] = None
    if 'y_full_st' not in st.session_state:
        st.session_state['y_full_st'] = None
    if 'roc_curve_data_xgb_st' not in st.session_state:
        st.session_state['roc_curve_data_xgb_st'] = None
    if 'xgb_model_loaded' not in st.session_state:
        st.session_state['xgb_model_loaded'] = False

    # Load pre-trained model components (XGBoost model, scaler, feature columns)
    xgb_model_loaded, scaler_loaded, feature_columns_loaded = load_model_components()

    # Check if model components loaded successfully
    if xgb_model_loaded is None or scaler_loaded is None or feature_columns_loaded is None:
        # load_model_components already displayed an error message if loading failed.
        st.stop() # Stop the app if model components failed to load

    # --- Data Loading and Preprocessing --- #
    st.subheader("1. Data Loading and Preprocessing")
    st.info("Loading and preprocessing data... This may take a moment.")
    X_full_processed_st, y_full_st = load_and_preprocess_data(scaler_loaded, feature_columns_loaded)

    # Explicitly check for None returns from load_and_preprocess_data in case of internal error
    if X_full_processed_st is None or y_full_st is None:
        # load_and_preprocess_data already displayed an error message if loading failed.
        st.stop() # Stop the app if data loading/preprocessing failed

    st.success("Data loaded and preprocessed successfully!")
    st.write(f"Shape of processed X_full: {X_full_processed_st.shape}")

    # --- Sidebar for Hyperparameter Controls (display only) --- #
    st.sidebar.header("XGBoost Model Parameters (Pre-trained)")
    st.sidebar.info("These parameters reflect the pre-trained model's configuration.")

    st.sidebar.slider(
        'XGBoost n_estimators', min_value=50, max_value=200, value=xgb_model_loaded.n_estimators, step=10, disabled=True
    )
    st.sidebar.slider(
        'XGBoost max_depth', min_value=3, max_value=10, value=xgb_model_loaded.max_depth, step=1, disabled=True
    )
    st.sidebar.slider(
        'XGBoost Learning Rate', min_value=0.001, max_value=0.5, value=float(xgb_model_loaded.learning_rate), format="%.3f", disabled=True
    )

    # --- Load and Evaluate Model Button --- #
    if st.sidebar.button('Load and Evaluate Model'):
        st.write("Loading pre-trained model and evaluating performance...")

        st.subheader("2. XGBoost Model Evaluation")
        st.info("Evaluating XGBoost model performance...")
        xgb_eval_results = evaluate_xgb_model(xgb_model_loaded, X_full_processed_st, y_full_st, XGB_OPTIMAL_THRESHOLD)
        st.success("XGBoost Model evaluation complete!")

        # Store results in session state for interactive section
        st.session_state['y_pred_proba_xgb_st'] = xgb_eval_results['y_pred_proba']
        st.session_state['y_full_st'] = y_full_st
        st.session_state['roc_curve_data_xgb_st'] = xgb_eval_results['roc_curve_data']
        st.session_state['xgb_model_loaded'] = True

        st.write(f"**Optimal Threshold for XGBoost**: {XGB_OPTIMAL_THRESHOLD:.2f}")
        st.write(f"**Precision (Fraud) with Optimal Threshold**: {xgb_eval_results['precision_optimized']:.4f}")
        st.write(f"**Recall (Fraud) with Optimal Threshold**: {xgb_eval_results['recall_optimized']:.4f}")
        st.write(f"**F1-Score (Fraud) with Optimal Threshold**: {xgb_eval_results['f1_optimized']:.4f}")
        st.write(f"**ROC-AUC**: {xgb_eval_results['roc_auc']:.4f}")

        st.write("**Confusion Matrix (Optimal Threshold)**")
        st.write(xgb_eval_results['conf_matrix'])

        st.write("**ROC Curve (with Optimal Threshold)**")
        fig_xgb_roc, ax_xgb_roc = plt.subplots(figsize=(8, 6))
        fpr_xgb, tpr_xgb, roc_auc_xgb = xgb_eval_results['roc_curve_data']
        ax_xgb_roc.plot(fpr_xgb, tpr_xgb, color='blue', label=f'ROC curve (area = {roc_auc_xgb:.2f})')

        # Plot the optimal threshold point on the ROC curve
        y_pred_thresholded_fixed = (xgb_eval_results['y_pred_proba'] > XGB_OPTIMAL_THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_full_st, y_pred_thresholded_fixed).ravel()
        fpr_at_opt_thresh = fp / (fp + tn)
        tpr_at_opt_thresh = tp / (tp + fn)
        ax_xgb_roc.plot(fpr_at_opt_thresh, tpr_at_opt_thresh, 'o', color='green', markersize=8, label=f'Optimal Thresh ({XGB_OPTIMAL_THRESHOLD:.2f})')

        ax_xgb_roc.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
        ax_xgb_roc.set_xlabel('False Positive Rate')
        ax_xgb_roc.set_ylabel('True Positive Rate')
        ax_xgb_roc.set_title('Receiver Operating Characteristic (ROC) Curve for XGBoost')
        ax_xgb_roc.legend()
        ax_xgb_roc.grid(True)
        st.pyplot(fig_xgb_roc)


        # --- Simple Risk Scoring (using XGBoost model's predictions) --- #
        st.header("3. Transaction Risk Scoring (using XGBoost model)")
        st.info("Converting XGBoost model's predicted probabilities into risk scores (0-100) and categorizing them.")

        # Convert predicted probabilities to risk scores (0-100)
        risk_scores = xgb_eval_results['y_pred_proba'] * 100

        # Categorize risk scores
        risk_categories = []
        for score in risk_scores:
            if score < RISK_MEDIUM_THRESHOLD_PERCENT:
                risk_categories.append('Low Risk')
            elif score >= RISK_MEDIUM_THRESHOLD_PERCENT and score < RISK_HIGH_THRESHOLD_PERCENT:
                risk_categories.append('Medium Risk')
            else:
                risk_categories.append('High Risk')

        risk_categories_series = pd.Series(risk_categories, name='Risk Category')

        st.subheader("Distribution of Risk Categories (on Full Dataset)")
        st.write("**Number of transactions per category:**")
        st.write(risk_categories_series.value_counts())
        st.write("**Percentage of transactions per category:**")
        st.write(risk_categories_series.value_counts(normalize=True) * 100)

        st.markdown("--- ")
        st.write("This application provides an interactive dashboard for UPI fraud detection, "
                 "using an XGBoost model and demonstrating a simple risk scoring mechanism.")

    # --- Interactive Threshold Optimization (XGBoost Model) --- #
    st.header("4. Interactive Threshold Optimization (XGBoost Model)")
    if st.session_state['xgb_model_loaded'] and st.session_state['y_pred_proba_xgb_st'] is not None:
        st.write("Adjust the slider to see how different classification thresholds affect the XGBoost model's performance metrics.")

        current_threshold_xgb = st.slider(
            'Select XGBoost Classification Threshold',
            min_value=0.0,
            max_value=1.0,
            value=XGB_OPTIMAL_THRESHOLD, # Default to the previously found optimal threshold
            step=0.01,
            key='xgb_threshold_slider'
        )

        # Recalculate metrics based on the current slider threshold
        y_pred_proba_xgb_session = st.session_state['y_pred_proba_xgb_st']
        y_full_session = st.session_state['y_full_st']
        fpr_xgb_session, tpr_xgb_session, roc_auc_xgb_session = st.session_state['roc_curve_data_xgb_st']

        y_pred_dynamic_xgb = (y_pred_proba_xgb_session > current_threshold_xgb).astype(int)

        precision_dynamic_xgb = precision_score(y_full_session, y_pred_dynamic_xgb, pos_label=1)
        recall_dynamic_xgb = recall_score(y_full_session, y_pred_dynamic_xgb, pos_label=1)
        f1_dynamic_xgb = f1_score(y_full_session, y_pred_dynamic_xgb, pos_label=1)
        conf_matrix_dynamic_xgb = confusion_matrix(y_full_session, y_pred_dynamic_xgb)

        st.write(f"**Metrics at Threshold {current_threshold_xgb:.2f}:**")
        st.write(f"Precision (Fraud): {precision_dynamic_xgb:.4f}")
        st.write(f"Recall (Fraud): {recall_dynamic_xgb:.4f}")
        st.write(f"F1-Score (Fraud): {f1_dynamic_xgb:.4f}")

        st.write("**Confusion Matrix:**")
        st.write(conf_matrix_dynamic_xgb)

        st.write("**ROC Curve with Current Threshold Point**")
        fig_xgb_roc_dynamic, ax_xgb_roc_dynamic = plt.subplots(figsize=(8, 6))
        ax_xgb_roc_dynamic.plot(fpr_xgb_session, tpr_xgb_session, color='blue', label=f'ROC curve (area = {roc_auc_xgb_session:.2f})')

        # Find the point on the ROC curve corresponding to the current_threshold
        tn_dyn, fp_dyn, fn_dyn, tp_dyn = confusion_matrix(y_full_session, y_pred_dynamic_xgb).ravel()
        fpr_at_current_thresh_dyn = fp_dyn / (fp_dyn + tn_dyn)
        tpr_at_current_thresh_dyn = tp_dyn / (tp_dyn + fn_dyn)
        ax_xgb_roc_dynamic.plot(fpr_at_current_thresh_dyn, tpr_at_current_thresh_dyn, 'o', color='green', markersize=8, label=f'Current Thresh ({current_threshold_xgb:.2f})')

        ax_xgb_roc_dynamic.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
        ax_xgb_roc_dynamic.set_xlabel('False Positive Rate')
        ax_xgb_roc_dynamic.set_ylabel('True Positive Rate')
        ax_xgb_roc_dynamic.set_title('Receiver Operating Characteristic (ROC) Curve for XGBoost')
        ax_xgb_roc_dynamic.legend()
        ax_xgb_roc_dynamic.grid(True)
        st.pyplot(fig_xgb_roc_dynamic)
    else:
        st.warning("Please load and evaluate the model first to enable interactive threshold optimization.")

# Run the Streamlit app
if __name__ == '__main__':
    streamlit_app()




