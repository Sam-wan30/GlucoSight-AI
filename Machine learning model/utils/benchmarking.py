"""Deterministic classical-model benchmarking for the active dataset."""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def benchmark_models(X, y, random_state=42):
    """Compare four classifiers on one deterministic stratified hold-out split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=random_state)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=random_state, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=random_state)),
        ]),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", SVC(probability=True, class_weight="balanced", random_state=random_state)),
        ]),
    }
    rows = []
    for name, estimator in models.items():
        try:
            estimator.fit(X_train, y_train)
            predicted = estimator.predict(X_test)
            probability = estimator.predict_proba(X_test)[:, 1]
            rows.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, predicted),
                "Precision": precision_score(y_test, predicted, zero_division=0),
                "Recall": recall_score(y_test, predicted, zero_division=0),
                "F1": f1_score(y_test, predicted, zero_division=0),
                "ROC AUC": roc_auc_score(y_test, probability),
            })
        except Exception as exc:
            rows.append({"Model": name, "Error": str(exc)})
    return pd.DataFrame(rows)

