from bioml_localization.features import AMINO_ACIDS, amino_acid_composition, clean_sequence


def test_clean_sequence_filters_non_canonical_chars():
    raw = "ACDZXXB*123"
    cleaned = clean_sequence(raw)
    assert cleaned == "ACD"


def test_amino_acid_composition_sums_to_one_for_non_empty_sequence():
    vector = amino_acid_composition("ACDEFGHIK")
    assert len(vector) == len(AMINO_ACIDS)
    assert abs(sum(vector) - 1.0) < 1e-9
