"""Inference functions for trained protein localization model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from bioml_localization.data import load_unlabeled_fasta
from bioml_localization.features import build_feature_matrix


def load_artifacts(artifacts_dir: str | Path):
    """Load persisted model artifacts from disk."""
    base = Path(artifacts_dir)
    model = joblib.load(base / "model.joblib")
    vectorizer = joblib.load(base / "vectorizer.joblib")
    label_encoder = joblib.load(base / "label_encoder.joblib")
    return model, vectorizer, label_encoder


def predict_dataframe(df: pd.DataFrame, artifacts_dir: str | Path) -> pd.DataFrame:
    """Run predictions for a dataframe with columns id and sequence."""
    required = {"id", "sequence"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input dataframe must include columns: {required}")

    model, vectorizer, label_encoder = load_artifacts(artifacts_dir)
    features = build_feature_matrix(df["sequence"].tolist(), vectorizer)

    pred_ids = model.predict(features)
    probabilities = model.predict_proba(features)
    confidence = probabilities.max(axis=1)
    labels = label_encoder.inverse_transform(pred_ids)

    return pd.DataFrame(
        {
            "id": df["id"].tolist(),
            "sequence": df["sequence"].tolist(),
            "predicted_label": labels,
            "confidence": confidence,
        }
    )


def predict_from_csv(csv_path: str | Path, artifacts_dir: str | Path) -> pd.DataFrame:
    """Predict localization labels from a CSV file."""
    df = pd.read_csv(csv_path)
    return predict_dataframe(df, artifacts_dir)


def predict_from_fasta(fasta_path: str | Path, artifacts_dir: str | Path) -> pd.DataFrame:
    """Predict localization labels from FASTA sequence file."""
    df = load_unlabeled_fasta(fasta_path)
    return predict_dataframe(df, artifacts_dir)
