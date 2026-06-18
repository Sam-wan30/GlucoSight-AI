"""SHAP explanations with a dependency-safe model-importance fallback."""

import numpy as np
import pandas as pd


def _positive_class_values(values):
    """Normalize SHAP output across common binary-classifier API shapes."""
    if isinstance(values, list):
        return np.asarray(values[-1])
    array = np.asarray(values)
    if array.ndim == 3:
        return array[:, :, -1]
    return array


def explain_prediction(model, processed_patient, processed_background, raw_patient, feature_names, fallback):
    """Return local/global explanation frames and the method used."""
    method = "Model contribution fallback"
    local_values = np.asarray(fallback, dtype=float)
    if hasattr(model, "feature_importances_"):
        global_values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        global_values = np.abs(np.asarray(model.coef_[0], dtype=float))
    else:
        global_values = np.abs(local_values)

    try:
        import shap

        background = np.asarray(processed_background)[:200]
        patient = np.asarray(processed_patient)
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, background)
        local_values = _positive_class_values(explainer.shap_values(patient))[0]
        background_values = _positive_class_values(explainer.shap_values(background))
        global_values = np.mean(np.abs(background_values), axis=0)
        method = "SHAP"
    except Exception:
        # Missing or incompatible SHAP should never prevent clinical inference.
        pass

    local = pd.DataFrame({
        "Feature": feature_names,
        "Value": [float(raw_patient[name]) for name in feature_names],
        "Contribution": np.asarray(local_values, dtype=float),
    }).sort_values("Contribution", key=abs, ascending=False)
    local["Direction"] = np.where(local["Contribution"] >= 0, "Increases risk", "Reduces risk")

    global_frame = pd.DataFrame({
        "Feature": feature_names,
        "Importance": np.asarray(global_values, dtype=float),
    }).sort_values("Importance", ascending=False)
    return local, global_frame, method

