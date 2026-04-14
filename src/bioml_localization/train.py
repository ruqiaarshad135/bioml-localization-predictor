"""Training entry points for the bioinformatics ML project."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from bioml_localization.data import load_labeled_csv
from bioml_localization.features import AMINO_ACIDS, build_feature_matrix, fit_tfidf_vectorizer
from bioml_localization.model import build_model, evaluate_model, top_feature_importance


def train_model(data_path: str | Path, output_dir: str | Path, test_size: float = 0.2, random_state: int = 42) -> dict:
    """Train localization model and persist artifacts + metrics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_labeled_csv(data_path)

    x_train, x_test, y_train_text, y_test_text = train_test_split(
        df["sequence"],
        df["label"],
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)
    y_test = label_encoder.transform(y_test_text)

    vectorizer = fit_tfidf_vectorizer(x_train.tolist())
    x_train_features = build_feature_matrix(x_train.tolist(), vectorizer)
    x_test_features = build_feature_matrix(x_test.tolist(), vectorizer)

    model = build_model(random_state=random_state)
    model.fit(x_train_features, y_train)

    labels = label_encoder.classes_.tolist()
    eval_result = evaluate_model(model, x_test_features, y_test, labels)

    feature_names = vectorizer.get_feature_names_out().tolist() + [f"aac_{aa}" for aa in AMINO_ACIDS]
    top_features = top_feature_importance(model, feature_names, top_k=20)

    artifacts = {
        "model": str(output_path / "model.joblib"),
        "vectorizer": str(output_path / "vectorizer.joblib"),
        "label_encoder": str(output_path / "label_encoder.joblib"),
        "metrics": str(output_path / "metrics.json"),
    }

    joblib.dump(model, artifacts["model"])
    joblib.dump(vectorizer, artifacts["vectorizer"])
    joblib.dump(label_encoder, artifacts["label_encoder"])

    metrics_payload = {
        "accuracy": eval_result.accuracy,
        "macro_f1": eval_result.macro_f1,
        "classes": labels,
        "classification_report": eval_result.report,
        "top_feature_importance": top_features,
        "dataset": {
            "source": str(data_path),
            "num_samples": int(df.shape[0]),
            "num_classes": int(df["label"].nunique()),
        },
    }

    with open(artifacts["metrics"], "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    return metrics_payload
