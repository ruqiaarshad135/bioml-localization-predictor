"""Command-line interface for the BioML Localization project."""

from __future__ import annotations

import argparse
import json

from bioml_localization.predict import predict_from_csv, predict_from_fasta
from bioml_localization.train import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bioml-localization",
        description="Bioinformatics ML pipeline for protein subcellular localization prediction.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train model and save artifacts")
    train_parser.add_argument("--data", required=True, help="Path to labeled CSV dataset")
    train_parser.add_argument("--out", default="artifacts", help="Output directory for artifacts")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    predict_parser = subparsers.add_parser("predict", help="Run inference using saved artifacts")
    predict_parser.add_argument("--artifacts", default="artifacts", help="Directory containing model files")
    predict_parser.add_argument("--input", required=True, help="Path to input CSV or FASTA")
    predict_parser.add_argument("--format", choices=["csv", "fasta"], required=True, help="Input file format")
    predict_parser.add_argument("--out", default="predictions.csv", help="Output predictions CSV")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        metrics = train_model(
            data_path=args.data,
            output_dir=args.out,
            test_size=args.test_size,
            random_state=args.seed,
        )
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "predict":
        if args.format == "csv":
            pred_df = predict_from_csv(args.input, args.artifacts)
        else:
            pred_df = predict_from_fasta(args.input, args.artifacts)
        pred_df.to_csv(args.out, index=False)
        print(f"Saved predictions to {args.out}")
        return

    parser.error("Unknown command")


if __name__ == "__main__":
    main()
