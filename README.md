# BioML Localization Predictor

This repository predicts **protein subcellular localization** (nucleus, membrane, cytoplasm) from amino-acid sequences using supervised ML.

It is designed to be:
- academically meaningful for final-year projects and supervision reviews,
- easy to run end-to-end,
- professionally structured for portfolio and GitHub presentation.

## Project Owner

- Developer: **Ruqia Arshad**
- GitHub: [ruqiaarshad135](https://github.com/ruqiaarshad135)

## Why This Project Is Strong For Bioinformatics

- Uses biological sequence data (protein FASTA/CSV) directly.
- Includes feature engineering commonly seen in computational biology:
  - amino-acid composition (AAC),
  - motif-sensitive character n-gram TF-IDF.
- Uses a supervised classifier with evaluation metrics and interpretability output.
- Supports reproducible training and prediction workflows.

## Repository Structure

```
.
├─ data/
│  └─ raw/
│     └─ protein_localization_sample.csv
├─ scripts/
│  └─ generate_synthetic_dataset.py
├─ src/
│  └─ bioml_localization/
│     ├─ cli.py
│     ├─ data.py
│     ├─ features.py
│     ├─ model.py
│     ├─ predict.py
│     └─ train.py
├─ tests/
│  ├─ test_features.py
│  └─ test_training.py
├─ .github/
│  ├─ ISSUE_TEMPLATE/
│  ├─ workflows/
│  └─ pull_request_template.md
├─ pyproject.toml
├─ requirements.txt
└─ README.md
```

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/ruqiaarshad135/bioml-localization-predictor.git
cd bioml-localization-predictor
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Train model

```bash
bioml-localization train --data data/raw/protein_localization_sample.csv --out artifacts
```

This command saves:
- `artifacts/model.joblib`
- `artifacts/vectorizer.joblib`
- `artifacts/label_encoder.joblib`
- `artifacts/metrics.json`

### 3. Predict from CSV

Input CSV requires:
- `id`
- `sequence`

```bash
bioml-localization predict --format csv --input data/raw/protein_localization_sample.csv --artifacts artifacts --out predictions.csv
```

### 4. Predict from FASTA

```bash
bioml-localization predict --format fasta --input examples/inference_sequences.fasta --artifacts artifacts --out predictions.csv
```

## Model Details

- Task: multiclass protein localization prediction.
- Algorithm: Random Forest (class-balanced).
- Features:
  - character TF-IDF ($n$-grams: 2 to 3),
  - 20-dimensional amino-acid composition.
- Metrics:
  - accuracy,
  - macro-F1,
  - per-class precision/recall/F1 report.

## Run Tests

```bash
pytest -q
```

## Re-generate Bigger Synthetic Dataset

```bash
python scripts/generate_synthetic_dataset.py
```

## Suggested Academic Extensions

1. Add real benchmark datasets (Swiss-Prot/DeepLoc subsets).
2. Compare Random Forest vs SVM vs XGBoost.
3. Add SHAP analysis for class-level explainability.
4. Add Flask/FastAPI web interface for live sequence upload.
5. Containerize + deploy on cloud for demo day.

## License

MIT
