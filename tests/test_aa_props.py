"""Tests for _aa_props: sequence encoding (property + OHE) used by the Python backend.

OHE tests run without R.  Property tests require immApex via rpy2 and use the
``require_ibex_r`` fixture (same pattern as test_ibex.py).
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# AMINO_ACIDS order (no R required)
# ---------------------------------------------------------------------------


def test_amino_acids_has_20_elements():
    from scibex._aa_props import AMINO_ACIDS

    assert len(AMINO_ACIDS) == 20


def test_amino_acids_order_matches_imm_apex():
    from scibex._aa_props import AMINO_ACIDS

    assert AMINO_ACIDS == list("ARNDCQEGHILKMFPSTWYV")


# ---------------------------------------------------------------------------
# get_lookup — OHE (no R required)
# ---------------------------------------------------------------------------


def test_get_lookup_ohe_non_exp_shape():
    from scibex._aa_props import get_lookup

    lut = get_lookup("OHE", is_exp=False)
    # 20 AAs + "." pad = 21 symbols; k = 21
    assert lut.shape == (21, 21)


def test_get_lookup_ohe_exp_shape():
    from scibex._aa_props import get_lookup

    lut = get_lookup("OHE", is_exp=True)
    # 20 AAs + "_" + "." = 22 symbols; k = 22
    assert lut.shape == (22, 22)


def test_get_lookup_ohe_is_identity_non_exp():
    from scibex._aa_props import get_lookup

    lut = get_lookup("OHE", is_exp=False)
    np.testing.assert_array_equal(lut, np.eye(21, dtype=np.float32))


def test_get_lookup_ohe_is_identity_exp():
    from scibex._aa_props import get_lookup

    lut = get_lookup("OHE", is_exp=True)
    np.testing.assert_array_equal(lut, np.eye(22, dtype=np.float32))


def test_get_lookup_unknown_encoding_raises():
    from scibex._aa_props import get_lookup

    with pytest.raises((ValueError, KeyError)):
        get_lookup("notAnEncoding", is_exp=False)


# ---------------------------------------------------------------------------
# get_lookup — property encodings (require immApex via rpy2)
# ---------------------------------------------------------------------------


def test_get_lookup_atchley_shape(require_ibex_r):
    from scibex._aa_props import get_lookup

    lut = get_lookup("atchleyFactors", is_exp=False)
    assert lut.shape == (21, 5)  # 20 AAs + pad row; 5 Atchley factors


def test_get_lookup_atchley_exp_shape(require_ibex_r):
    from scibex._aa_props import get_lookup

    lut = get_lookup("atchleyFactors", is_exp=True)
    assert lut.shape == (22, 5)  # 20 AAs + "_" + pad; 5 cols


def test_get_lookup_pad_row_is_zeros_property(require_ibex_r):
    from scibex._aa_props import get_lookup

    lut = get_lookup("atchleyFactors", is_exp=False)
    assert np.all(lut[-1] == 0.0)


def test_get_lookup_exp_underscore_and_pad_rows_are_zeros(require_ibex_r):
    from scibex._aa_props import get_lookup

    lut = get_lookup("atchleyFactors", is_exp=True)
    assert np.all(lut[20] == 0.0)  # "_" separator → no properties
    assert np.all(lut[21] == 0.0)  # "." pad → zeros


def test_get_lookup_atchley_alanine_values_match_imm_apex(require_ibex_r):
    """Row 0 (A) must match immApex's propertyEncoder values for Alanine."""
    from scibex._aa_props import get_lookup

    lut = get_lookup("atchleyFactors", is_exp=False)
    expected = np.array([-0.591, -1.302, -0.733, 1.570, -0.146], dtype=np.float32)
    np.testing.assert_allclose(lut[0], expected, atol=1e-3)


def test_get_lookup_cruciani_shape(require_ibex_r):
    from scibex._aa_props import get_lookup

    assert get_lookup("crucianiProperties", is_exp=False).shape == (21, 3)


def test_get_lookup_kidera_shape(require_ibex_r):
    from scibex._aa_props import get_lookup

    assert get_lookup("kideraFactors", is_exp=False).shape == (21, 10)


def test_get_lookup_mswhim_shape(require_ibex_r):
    from scibex._aa_props import get_lookup

    assert get_lookup("MSWHIM", is_exp=False).shape == (21, 3)


def test_get_lookup_tscales_shape(require_ibex_r):
    from scibex._aa_props import get_lookup

    assert get_lookup("tScales", is_exp=False).shape == (21, 5)


# ---------------------------------------------------------------------------
# encode_sequences — shape (OHE, no R required)
# ---------------------------------------------------------------------------


def test_encode_sequences_ohe_shape_cdr3():
    from scibex._aa_props import encode_sequences

    x = encode_sequences(["CARDYW"], "OHE", is_exp=False)
    assert x.shape == (1, 45 * 21)


def test_encode_sequences_ohe_shape_exp():
    from scibex._aa_props import encode_sequences

    x = encode_sequences(["GFTFS-ISGSGGST-CARDYW"], "OHE", is_exp=True)
    assert x.shape == (1, 90 * 22)


def test_encode_sequences_returns_float32_ohe():
    from scibex._aa_props import encode_sequences

    x = encode_sequences(["CARDYW"], "OHE", is_exp=False)
    assert x.dtype == np.float32


# ---------------------------------------------------------------------------
# encode_sequences — OHE values (no R required)
# ---------------------------------------------------------------------------


def test_encode_sequences_ohe_alanine_first_position():
    """'A' at position 0 → one-hot at index 0."""
    from scibex._aa_props import encode_sequences

    x = encode_sequences(["A"], "OHE", is_exp=False)
    expected = np.zeros(21, dtype=np.float32)
    expected[0] = 1.0
    np.testing.assert_array_equal(x[0, :21], expected)


def test_encode_sequences_ohe_pad_position():
    """Padding positions → one-hot for '.' (last column, index 20)."""
    from scibex._aa_props import encode_sequences

    x = encode_sequences(["A"], "OHE", is_exp=False)
    # Position 1 is padding (sequence length 1, max_length 45)
    pad_slot = x[0, 21:42]
    expected = np.zeros(21, dtype=np.float32)
    expected[20] = 1.0
    np.testing.assert_array_equal(pad_slot, expected)


def test_encode_sequences_exp_separator_ohe_has_own_column():
    """'-' separator in EXP OHE → one-hot at index 20 ('_' column)."""
    from scibex._aa_props import encode_sequences

    # "A-R": position 0=A (idx 0), position 1='-' (idx 20), position 2=R (idx 1)
    x = encode_sequences(["A-R"], "OHE", is_exp=True)
    sep_slot = x[0, 22:44]  # second position, k=22
    expected = np.zeros(22, dtype=np.float32)
    expected[20] = 1.0  # '_' is at index 20
    np.testing.assert_array_equal(sep_slot, expected)


def test_encode_sequences_truncates_long_sequence():
    from scibex._aa_props import encode_sequences

    x = encode_sequences(["A" * 60], "OHE", is_exp=False)
    assert x.shape == (1, 45 * 21)


def test_encode_sequences_ohe_multiple_sequences_independent():
    from scibex._aa_props import encode_sequences

    x = encode_sequences(["A", "R"], "OHE", is_exp=False)
    assert x.shape == (2, 45 * 21)
    # A → one-hot index 0
    np.testing.assert_array_equal(x[0, 0], 1.0)
    np.testing.assert_array_equal(x[0, 1:21], np.zeros(20))
    # R → one-hot index 1
    np.testing.assert_array_equal(x[1, 1], 1.0)
    np.testing.assert_array_equal(x[1, 0], 0.0)


# ---------------------------------------------------------------------------
# encode_sequences — property (require R)
# ---------------------------------------------------------------------------


def test_encode_sequences_property_shape_cdr3(require_ibex_r):
    from scibex._aa_props import encode_sequences

    x = encode_sequences(["CARDYW", "CARDSSGYW"], "atchleyFactors", is_exp=False)
    assert x.shape == (2, 45 * 5)


def test_encode_sequences_property_shape_exp(require_ibex_r):
    from scibex._aa_props import encode_sequences

    x = encode_sequences(["GFTFS-ISGSGGST-CARDYW"], "atchleyFactors", is_exp=True)
    assert x.shape == (1, 90 * 5)


def test_encode_sequences_returns_float32_property(require_ibex_r):
    from scibex._aa_props import encode_sequences

    x = encode_sequences(["CARDYW"], "atchleyFactors", is_exp=False)
    assert x.dtype == np.float32


def test_encode_sequences_alanine_property_first_position(require_ibex_r):
    """'A' at position 0 → atchleyFactors for Alanine (from immApex)."""
    from scibex._aa_props import encode_sequences

    x = encode_sequences(["A"], "atchleyFactors", is_exp=False)
    expected = np.array([-0.591, -1.302, -0.733, 1.570, -0.146], dtype=np.float32)
    np.testing.assert_allclose(x[0, :5], expected, atol=1e-3)


def test_encode_sequences_padding_is_zeros_property(require_ibex_r):
    """Pad positions (beyond sequence end) → zero in property mode."""
    from scibex._aa_props import encode_sequences

    x = encode_sequences(["A"], "atchleyFactors", is_exp=False)
    assert np.all(x[0, 5:] == 0.0)


def test_encode_sequences_exp_separator_property_is_zero(require_ibex_r):
    """'-' separator in EXP property encoding → zero vector."""
    from scibex._aa_props import encode_sequences

    # "A-R": position 1 (the separator) should be all zeros
    x = encode_sequences(["A-R"], "atchleyFactors", is_exp=True)
    sep_slot = x[0, 5:10]  # second position, k=5
    np.testing.assert_array_equal(sep_slot, np.zeros(5, dtype=np.float32))


def test_encode_sequences_parity_with_immapex(require_ibex_r):
    """Python encode_sequences must match immApex::propertyEncoder output exactly."""
    import rpy2.robjects as ro

    from scibex._aa_props import encode_sequences

    seqs = ["CARDYW", "CARDSSGYW", "CARDTGYW"]
    py_out = encode_sequences(seqs, "atchleyFactors", is_exp=False)

    # Get the R ground truth
    r_seqs = "c(" + ",".join(f"'{s}'" for s in seqs) + ")"
    cmd = (
        f"immApex::propertyEncoder({r_seqs}, property.set='atchleyFactors',"
        f" max.length=45, convert.to.matrix=TRUE, verbose=FALSE)"
    )
    result = ro.r(cmd)
    r_out = np.array(result.rx2("flattened"), dtype=np.float32)  # type: ignore

    np.testing.assert_allclose(py_out, r_out, atol=1e-5)
