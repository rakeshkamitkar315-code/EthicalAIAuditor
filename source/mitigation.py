# ===============================================
# Fairness Mitigation Example: Adult Dataset
# Pre-processing, In-processing, Post-processing
# ===============================================

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score

# ---------------------------
# 1. Load and preprocess dataset
# ---------------------------
data_path = r"C:\Users\rakes\OneDrive\Desktop\EthicalAIAuditor\data\adult.csv"
df = pd.read_csv(data_path)

# Convert target (income) to numeric
df['income'] = df['income'].apply(lambda x: 1 if str(x).strip() == '>50K' else 0)

# Define target, sensitive attribute, and features
y = df['income'].values
S = df['gender'].values
X = df.drop(columns=['income', 'gender'])

# One-Hot Encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = pd.DataFrame(
    encoder.fit_transform(X[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols)
)
X_numeric = X.drop(columns=categorical_cols).reset_index(drop=True)
X_final = pd.concat([X_numeric, X_encoded], axis=1)


warnings.filterwarnings("ignore", category=FutureWarning)

# Scale features for better convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# Split dataset
X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(
    X_scaled, y, S, test_size=0.2, random_state=42
)

print("Unique values in sensitive attribute:", np.unique(S_train))

# ============================
# 3. Pre-processing Mitigation
# ============================
def resample_minority(X, y, sensitive):
    df_copy = pd.DataFrame(X)
    df_copy['y'] = y
    df_copy['sensitive'] = sensitive

    values, counts = np.unique(sensitive, return_counts=True)
    if len(values) < 2:
        print("Warning: Only one group in sensitive attribute. Skipping pre-processing.")
        return X, y, sensitive

    minority_value = values[np.argmin(counts)]
    majority_value = values[np.argmax(counts)]

    minority = df_copy[df_copy['sensitive'] == minority_value]
    majority = df_copy[df_copy['sensitive'] == majority_value]

    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=42
    )

    new_df = pd.concat([majority, minority_upsampled])
    y_new = new_df['y'].values
    S_new = new_df['sensitive'].values
    X_new = new_df.drop(columns=['y', 'sensitive']).values
    return X_new, y_new, S_new

X_train_pre, y_train_pre, S_train_pre = resample_minority(X_train, y_train, S_train)

pre_model = LogisticRegression(max_iter=2000)
pre_model.fit(X_train_pre, y_train_pre)
preds_pre = pre_model.predict(X_test)

# ============================
# 4. In-processing Mitigation
# ============================
est = LogisticRegression(max_iter=2000)
mitigator = ExponentiatedGradient(est, constraints=DemographicParity())
mitigator.fit(X_train, y_train, sensitive_features=S_train)
preds_in = mitigator.predict(X_test)

# ============================
# 5. Post-processing Mitigation
# ============================
postprocessor = ThresholdOptimizer(
    estimator=LogisticRegression(max_iter=2000),
    constraints="demographic_parity",
    predict_method="predict_proba"
)
postprocessor.fit(X_train, y_train, sensitive_features=S_train)
preds_post = postprocessor.predict(X_test, sensitive_features=S_test)

# ============================
# 6. Evaluation
# ============================
print("\n===== Accuracy Scores =====")
print("Pre-processing:", accuracy_score(y_test, preds_pre))
print("In-processing:", accuracy_score(y_test, preds_in))
print("Post-processing:", accuracy_score(y_test, preds_post))

print("\n===== Fairness Metrics =====")
print("Demographic Parity Difference:")
print("Pre-processing:", demographic_parity_difference(y_test, preds_pre, sensitive_features=S_test))
print("In-processing:", demographic_parity_difference(y_test, preds_in, sensitive_features=S_test))
print("Post-processing:", demographic_parity_difference(y_test, preds_post, sensitive_features=S_test))

print("\nEqualized Odds Difference:")
print("Pre-processing:", equalized_odds_difference(y_test, preds_pre, sensitive_features=S_test))
print("In-processing:", equalized_odds_difference(y_test, preds_in, sensitive_features=S_test))
print("Post-processing:", equalized_odds_difference(y_test, preds_post, sensitive_features=S_test))

methods = ["Pre-processing", "In-processing", "Post-processing"]
accuracy = [0.8566, 0.8390, 0.8375]
dp_diff = [0.1778, 0.0161, 0.0058]

fig, ax1 = plt.subplots(figsize=(7,4))
ax2 = ax1.twinx()

ax1.bar(methods, accuracy, alpha=0.6, label="Accuracy")
ax2.plot(methods, dp_diff, color='red', marker='o', label="Demographic Parity Diff")

ax1.set_ylabel("Accuracy", color='blue')
ax2.set_ylabel("Fairness (DP Diff)", color='red')
ax1.set_title("Fairness - Accuracy Tradeoff")
ax1.grid(True)

plt.show()