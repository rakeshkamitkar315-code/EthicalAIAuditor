import streamlit as st
import sys
import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import warnings
import numpy as np
# Import necessary components from your source package
from source.preprocess import preprocess
from source._utils import load_data_and_preprocess
from source.fairness_check import demographic_parity_difference

# --- Path Fix for Module Discovery ---
# This ensures Python can find the 'source' package when run via 'streamlit run'.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.insert(0, project_root)
# -------------------------------------

st.title("CognitiveAI - An AI Model Bias Detector")

# Load model and data using the project_root path for robustness
model_path = os.path.join(project_root, 'models', 'logreg.pkl')
data_path = os.path.join(project_root, 'data', 'adult.csv')

try:
    clf = joblib.load(model_path)
    df = pd.read_csv(data_path)
except FileNotFoundError as e:
    st.error(f"Error loading required files. Please ensure the 'models' and 'data' folders exist in the project root: {e}")
    st.stop()

# Preprocess data
X_train, X_test, y_train, y_test, S_train, S_test = preprocess(df)

# Ignore unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

st.header("Model Evaluation")
# Show model accuracy
st.write("Our Model Accuracy:", clf.score(X_test, y_test))

# Show demographic parity difference
preds = clf.predict(X_test)
st.write("Model's Demographic parity diff:", demographic_parity_difference(preds, S_test))

# SHAP feature explanation block
st.header("Feature Contribution Analysis (SHAP)")
if st.button("Show SHAP (SHapley Additive exPlanations) top features"):
    
    with st.spinner('Calculating and aggregating SHAP values...'):
        explainer = shap.Explainer(clf, X_train)
        # Use a smaller sample for fast calculation in the dashboard
        X_sample = X_test.iloc[:500] 
        shap_values_raw = explainer(X_sample)
        
        # --- Aggregation Logic (The Core Fix) ---
        
        # 1. Get the absolute SHAP values for aggregation
        abs_shap_vals = np.abs(shap_values_raw.values).mean(axis=0)
        
        # 2. Map one-hot encoded columns back to their original feature names
        feature_names = X_train.columns
        aggregated_shap = {}
        
        for name, value in zip(feature_names, abs_shap_vals):
            # Split the column name to find the original feature name
            # e.g., 'Marital-Status_Married-civilian-spouse' -> 'Marital-Status'
            original_feature = name.split('_')[0] 
            
            # For non-encoded columns (e.g., Age, Education-Num), the split won't change the name
            if '_' not in name:
                original_feature = name
            
            # Sum up the absolute SHAP values for the original feature
            if original_feature not in aggregated_shap:
                aggregated_shap[original_feature] = 0
            
            aggregated_shap[original_feature] += value

        # 3. Convert the aggregated dictionary to a sorted Series for plotting
        aggregated_series = pd.Series(aggregated_shap).sort_values(ascending=False)
        
        # 4. Create the Plot
        fig, ax = plt.subplots(figsize=(10, len(aggregated_series) * 0.5))
        aggregated_series.plot(kind='barh', ax=ax, color='#007ACC')
        ax.set_title("Aggregated Feature Importance (Mean Absolute SHAP Value)", fontsize=14)
        ax.set_xlabel("Average Magnitude of Impact on Model Output", fontsize=12)
        ax.set_ylabel("Original Feature", fontsize=12)
        ax.invert_yaxis() # Highest impact feature on top
        plt.tight_layout()
        
        # 5. Display in Streamlit
        st.pyplot(fig)

# Example scatter plot (Keeping your original example)
st.header("Example Plot")
fig, ax = plt.subplots()
ax.scatter([1, 2, 3], [1, 2, 3])
ax.set_title("Example Scatter Plot")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
st.pyplot(fig)
