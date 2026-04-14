"""Model setup and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score


@dataclass
class EvalResult:
    """Container for metrics and detailed class-wise report."""

    accuracy: float
    macro_f1: float
    report: dict


def build_model(random_state: int = 42) -> RandomForestClassifier:
    """Create the baseline model for protein localization classification."""
    return RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def evaluate_model(model: RandomForestClassifier, x_test, y_test, labels: list[str]) -> EvalResult:
    """Compute headline and detailed evaluation metrics."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(
        y_test,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return EvalResult(
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        report=report,
    )


def top_feature_importance(model: RandomForestClassifier, feature_names: list[str], top_k: int = 20) -> list[dict]:
    """Return top-k global feature importance values for interpretability."""
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_k]
    return [
        {"feature": feature_names[idx], "importance": float(importances[idx])}
        for idx in order
    ]
