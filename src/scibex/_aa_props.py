"""Sequence encoding for the Python backend, parity with immApex.

Property matrices are fetched from immApex::sequenceEncoder via rpy2 on first
use and cached.  OHE is computed locally (it's just an identity matrix).

Encoding algorithm mirrors Ibex.training/ibex_train/data.py:
  tokens [N, L] → lookup[tokens] → [N, L, k] → flatten → [N, L*k]
"""

from __future__ import annotations

from functools import cache

import numpy as np

AMINO_ACIDS: list[str] = list("ARNDCQEGHILKMFPSTWYV")

MAX_LENGTH_CDR3 = 45  # non-EXP models (CDR3 only)
MAX_LENGTH_EXP = 90  # EXP models (CDR1 + CDR2 + CDR3)

_VALID_PROPERTY_ENCODINGS = frozenset(
    [
        "atchleyFactors",
        "crucianiProperties",
        "kideraFactors",
        "MSWHIM",
        "tScales",
    ]
)


@cache
def _immapex_property_matrix(encoding: str) -> np.ndarray:
    """Return [20, k] property lookup from immApex::sequenceEncoder via rpy2.

    Calls sequenceEncoder on the 20 canonical amino acids with max.length=1 so
    the [N=20, L*k = k] flattened output is exactly one row per residue.
    Uses importr("immApex").sequenceEncoder so rpy2's SignatureTranslatedFunction
    converts kwargs (property_set → property.set etc.) without string eval.
    Result is cached per encoding for the lifetime of the process.
    """
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    immapex = importr("immApex")
    aa_vec = ro.StrVector(AMINO_ACIDS)
    result = immapex.sequenceEncoder(
        aa_vec,
        mode="property",
        property_set=encoding,
        max_length=1,
        convert_to_matrix=True,
        verbose=False,
    )
    return np.array(result.rx2("flattened"), dtype=np.float32)  # [20, k]


def get_lookup(encoding: str, *, is_exp: bool) -> np.ndarray:
    """Return lookup table [V, k] mapping residue index → property/OHE vector.

    V = 21 (non-EXP: 20 AAs + '.' pad) or 22 (EXP: 20 AAs + '_' + '.' pad).
    For OHE k = V (identity matrix).  For property encodings k = # properties;
    unknown/pad rows are all zeros.
    """
    extra = 1 if is_exp else 0  # extra row for the '_' CDR separator

    if encoding == "OHE":
        n = 20 + extra + 1
        return np.eye(n, dtype=np.float32)

    if encoding not in _VALID_PROPERTY_ENCODINGS:
        raise ValueError(
            f"Unknown encoding {encoding!r}. Valid property encodings: {sorted(_VALID_PROPERTY_ENCODINGS)}"
        )

    prop = _immapex_property_matrix(encoding)  # [20, k]
    pad_rows = np.zeros((extra + 1, prop.shape[1]), dtype=np.float32)
    return np.vstack([prop, pad_rows])  # [20 + extra + 1, k]


def encode_sequences(
    seqs: list[str],
    encoding: str,
    *,
    is_exp: bool,
) -> np.ndarray:
    """Encode sequences to [N, L*k] matching immApex's flattened output.

    Sequences longer than max_length are truncated; shorter ones are padded
    with the pad-symbol row (zero-vector for property, one-hot '.' for OHE).
    For EXP models, '-' (the CDR1-CDR2-CDR3 separator) is treated as '_'.
    """
    max_length = MAX_LENGTH_EXP if is_exp else MAX_LENGTH_CDR3
    lut = get_lookup(encoding, is_exp=is_exp)
    k = lut.shape[1]

    # Vocab: AMINO_ACIDS at 0-based indices 0..19.
    # EXP: '_' / '-' separator → index 20; pad '.' → index 21.
    # Non-EXP: pad '.' → index 20.
    vocab: dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    if is_exp:
        vocab["_"] = 20
        vocab["-"] = 20  # raw CDR1-CDR2-CDR3 separator before normalisation
    pad_idx = 20 + (1 if is_exp else 0)

    N = len(seqs)
    result = np.empty((N, max_length * k), dtype=np.float32)
    for i, seq in enumerate(seqs):
        seq = seq[:max_length]
        tokens = np.full(max_length, pad_idx, dtype=np.int32)
        for j, ch in enumerate(seq):
            tokens[j] = vocab.get(ch, pad_idx)
        result[i] = lut[tokens].reshape(-1)
    return result
