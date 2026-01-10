from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_model(X_train, y_train, model_name="logreg"):
    if model_name == "logreg":
        model = LogisticRegression(max_iter=2000, n_jobs=-1)
    elif model_name == "svm":
        model = LinearSVC()
    else:
        raise ValueError("Unsupported model name")

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return accuracy_score(y_test, y_pred)
