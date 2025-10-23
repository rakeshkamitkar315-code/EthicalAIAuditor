import os
import warnings
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from ._utils import load_data_and_preprocess  

def train_model(X_train, y_train):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf

def evaluate(clf, X_test, y_test):
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    return preds

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=UserWarning)

    # Load and preprocess data
    X_train, X_test, y_train, y_test, S_train, S_test = load_data_and_preprocess()
    
    # Train model
    clf = train_model(X_train, y_train)
    
    # Ensure models folder exists
    os.makedirs("models", exist_ok=True)
    
    # Save trained model
    joblib.dump(clf, "models/logreg.pkl")
    
    # Evaluate
    evaluate(clf, X_test, y_test)
