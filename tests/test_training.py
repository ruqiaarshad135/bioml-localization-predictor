from pathlib import Path

from bioml_localization.train import train_model


def test_train_model_creates_expected_artifacts(tmp_path: Path):
    data_path = Path("data/raw/protein_localization_sample.csv")
    out_dir = tmp_path / "artifacts"

    metrics = train_model(data_path=data_path, output_dir=out_dir, test_size=0.3, random_state=7)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert (out_dir / "model.joblib").exists()
    assert (out_dir / "vectorizer.joblib").exists()
    assert (out_dir / "label_encoder.joblib").exists()
    assert (out_dir / "metrics.json").exists()
