import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import joblib

# Import preprocess function depending on context
try:
    from source.preprocess import preprocess  # for package execution
except ImportError:
    from preprocess import preprocess  # for direct script execution

# Fairness metrics
def demographic_parity_difference(y_pred, sensitive):
    """Compute demographic parity difference between privileged and unprivileged groups."""
    y_pred = np.array(y_pred)
    sensitive = np.array(sensitive)
    p_priv = y_pred[sensitive == 1].mean()
    p_unpriv = y_pred[sensitive == 0].mean()
    return p_priv - p_unpriv

def equality_of_opportunity_difference(y_true, y_pred, sensitive):
    """Compute equality of opportunity difference (TPR difference)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sensitive = np.array(sensitive)

    def tpr(y_true_subset, y_pred_subset):
        tn, fp, fn, tp = confusion_matrix(y_true_subset, y_pred_subset).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    tpr_priv = tpr(y_true[sensitive == 1], y_pred[sensitive == 1])
    tpr_unpriv = tpr(y_true[sensitive == 0], y_pred[sensitive == 0])
    return tpr_priv - tpr_unpriv

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    df_path = os.path.join('data', 'adult.csv')
    df = pd.read_csv(df_path).replace('?', pd.NA).dropna()
    X_train, X_test, y_train, y_test, S_train, S_test = preprocess(df)

    # Load trained model
    clf_path = os.path.join('models', 'logreg.pkl')
    clf = joblib.load(clf_path)

    # Make predictions
    preds = clf.predict(X_test)

    # Compute custom fairness metrics
    dp_diff = demographic_parity_difference(preds, S_test)
    eo_diff = equality_of_opportunity_difference(y_test, preds, S_test)
    print("Demographic parity diff:", dp_diff)
    print("Equality of opportunity diff:", eo_diff)

    # Compute fairness metrics using Fairlearn
    from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate

    metrics = {
        'accuracy': lambda y_true, y_pred: (y_true == y_pred).mean(),
        'selection_rate': selection_rate,
        'tpr': true_positive_rate
    }

    mf = MetricFrame(metrics=metrics, y_true=y_test, y_pred=preds, sensitive_features=S_test)
    print("\nMetrics by group:\n", mf.by_group)
    print("\nOverall metrics:\n", mf.overall)
