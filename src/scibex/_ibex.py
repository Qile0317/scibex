from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import StrVector, pandas2ri

from ._r import ibex_r


def _raw_embed(
    sequences: list[str],
    chain: Literal["Heavy", "Light"],
    method: Literal["encoder", "geometric"],
    encoder_model: Literal["CNN", "VAE", "CNN.EXP", "VAE.EXP"],
    encoder_input: Literal["atchleyFactors", "crucianiProperties", "kideraFactors", "MSWHIM", "tScales", "OHE"],
    geometric_theta: float,
    species: Literal["Human", "Mouse"],
    verbose: bool,
) -> np.ndarray:
    """Call R's Ibex_matrix with a clean list of strings. No None handling."""
    r_result = ibex_r().Ibex_matrix(
        StrVector(sequences),
        chain=chain,
        method=method,
        encoder_model=encoder_model,
        encoder_input=encoder_input,
        geometric_theta=geometric_theta,
        species=species,
        verbose=verbose,
    )
    with (ro.default_converter + pandas2ri.converter).context():
        df = ro.conversion.get_conversion().rpy2py(r_result)
    return df.to_numpy()


def ibex_matrix(
    sequences: Sequence[str | None],
    chain: Literal["Heavy", "Light"] = "Heavy",
    method: Literal["encoder", "geometric"] = "encoder",
    encoder_model: Literal["CNN", "VAE", "CNN.EXP", "VAE.EXP"] = "VAE",
    encoder_input: Literal[
        "atchleyFactors", "crucianiProperties", "kideraFactors", "MSWHIM", "tScales", "OHE"
    ] = "atchleyFactors",
    geometric_theta: float = 3.14159265,
    species: Literal["Human", "Mouse"] = "Human",
    verbose: bool = False,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Embed BCR sequences using the Ibex R package.

    Returns an [N, D] float array aligned to the input sequence list.  ``None``
    entries (missing sequences) receive rows filled with ``fill_value`` (default
    ``0.0``; pass ``float("nan")`` for NaN).  When ``verbose=True`` a warning is
    emitted if any sequences are ``None``.
    """
    valid_indices = [i for i, s in enumerate(sequences) if s is not None]
    valid_seqs = [s for s in sequences if s is not None]

    if not valid_seqs:
        raise ValueError("All sequences are None; cannot determine embedding dimension.")

    embedding = _raw_embed(
        valid_seqs,
        chain=chain,
        method=method,
        encoder_model=encoder_model,
        encoder_input=encoder_input,
        geometric_theta=geometric_theta,
        species=species,
        verbose=verbose,
    )

    full = np.full((len(sequences), embedding.shape[1]), fill_value)
    for out_i, seq_i in enumerate(valid_indices):
        full[seq_i] = embedding[out_i]

    if verbose:
        n_missing = len(sequences) - len(valid_indices)
        if n_missing:
            warnings.warn(
                f"ibex_matrix: {n_missing} of {len(sequences)} sequences are None; rows filled with {fill_value}.",
                UserWarning,
                stacklevel=2,
            )

    return full
