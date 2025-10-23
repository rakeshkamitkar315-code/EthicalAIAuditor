import numpy as np
import shap
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .preprocess import preprocess  # assuming your folder is named 'src'

def explain_model(clf, X_train_scaled, X_sample_scaled):
    # Ensure input arrays are float64 numpy arrays
    X_train_array = np.array(X_train_scaled, dtype=np.float64)
    X_sample_array = np.array(X_sample_scaled, dtype=np.float64)

    # ✅ Just pass model and background data — no 'masker' argument here
    explainer = shap.LinearExplainer(clf, X_train_array)
    shap_values = explainer(X_sample_array)

    # Convert shap values to float64 array to prevent numpy 'rint' errors
    shap_values_array = shap_values.values 

    # Show SHAP summary plot
    shap.summary_plot(shap_values_array, X_sample_array)

    return shap_values_array


if __name__ == "__main__":
    # Load and preprocess dataset
    df = pd.read_csv('data/adult.csv')
    X_train, X_test, y_train, y_test, S_train, S_test = preprocess(df)
    print(X_train.shape, X_test.shape)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression model
    clf = LogisticRegression(solver='lbfgs', max_iter=3000)
    clf.fit(X_train_scaled, y_train)

    # Explain predictions on first 100 samples
    shap_values = explain_model(clf, X_train_scaled, X_test_scaled[:100])