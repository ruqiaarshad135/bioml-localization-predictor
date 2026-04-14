"""Data loading utilities for sequence classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = ("id", "sequence", "label")


@dataclass(frozen=True)
class SequenceRecord:
    """Simple in-memory representation of a labeled protein sequence."""

    record_id: str
    sequence: str
    label: str


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that the dataset contains expected columns and values."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    if df["sequence"].isna().any() or (df["sequence"].str.len() == 0).any():
        raise ValueError("Found empty sequences in dataset")

    if df["label"].nunique() < 2:
        raise ValueError("Dataset needs at least 2 target classes")


def load_labeled_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate labeled sequences from a CSV file."""
    df = pd.read_csv(path)
    validate_dataframe(df)
    return df


def parse_fasta_lines(lines: Iterable[str]) -> list[tuple[str, str]]:
    """Parse FASTA content into a list of (id, sequence)."""
    records: list[tuple[str, str]] = []
    current_id: str | None = None
    current_seq: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                records.append((current_id, "".join(current_seq)))
            current_id = line[1:].strip() or f"seq_{len(records) + 1}"
            current_seq = []
        else:
            current_seq.append(line.upper())

    if current_id is not None:
        records.append((current_id, "".join(current_seq)))

    if not records:
        raise ValueError("No FASTA records were found")
    return records


def load_unlabeled_fasta(path: str | Path) -> pd.DataFrame:
    """Load unlabeled FASTA sequences for inference."""
    with open(path, "r", encoding="utf-8") as handle:
        records = parse_fasta_lines(handle.readlines())

    return pd.DataFrame(
        {"id": [rec_id for rec_id, _ in records], "sequence": [seq for _, seq in records]}
    )
