import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def preprocess(df, sensitive='gender'):
    # Drop rows with missing values
    df = df.replace('?', pd.NA).dropna()

    # Binary target: income >50K
    df['target'] = (df['income'].str.strip() == '>50K').astype(int)

    # Keep sensitive attribute
    sensitive_series = df[sensitive].str.strip().copy()

    # Drop original target and sensitive from X
    X = df.drop(columns=['income', 'target', sensitive])

    # One-hot encode categorical features
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    y = df['target'].values
    S = (sensitive_series == 'Male').astype(int)

    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(
        X_enc, y, S, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test, S_train, S_test

if __name__ == "__main__":
    df = pd.read_csv('data/adult.csv')
    X_train, X_test, y_train, y_test, S_train, S_test = preprocess(df)
    print(X_train.shape, X_test.shape)
