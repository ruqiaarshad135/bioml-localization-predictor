"""Generate synthetic protein sequences for localization classes.

This script creates an educational dataset for demonstrating a complete ML pipeline.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


def _build_sequence(length: int, alphabet: str, motif: str, motif_repeat: int) -> str:
    seq = "".join(random.choice(alphabet) for _ in range(length - len(motif) * motif_repeat))
    return seq + (motif * motif_repeat)


def generate_dataset(n_per_class: int = 60, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)

    rows = []
    for i in range(n_per_class):
        rows.append(
            {
                "id": f"nucleus_{i+1}",
                "sequence": _build_sequence(120, "KRQNASTPDEG", "KKRK", 3),
                "label": "nucleus",
            }
        )
        rows.append(
            {
                "id": f"membrane_{i+1}",
                "sequence": _build_sequence(120, "LIVAFWMGSTCNQ", "LVVAI", 3),
                "label": "membrane",
            }
        )
        rows.append(
            {
                "id": f"cytoplasm_{i+1}",
                "sequence": _build_sequence(120, "ADEGSTNPQHCMR", "DEDE", 3),
                "label": "cytoplasm",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    df = generate_dataset(n_per_class=60, seed=42)
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "protein_localization_synthetic.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} samples to {out_path}")


if __name__ == "__main__":
    main()
