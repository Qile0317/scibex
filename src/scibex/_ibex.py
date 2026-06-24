from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import StrVector, pandas2ri

from ._r import ibex_r
from ._types import Backend, Chain, EncoderInput, EncoderModel, Method, Species


def _raw_embed(
    sequences: list[str],
    chain: Chain,
    method: Method,
    encoder_model: EncoderModel,
    encoder_input: EncoderInput,
    geometric_theta: float,
    species: Species,
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


def _raw_embed_python(
    sequences: list[str],
    chain: Chain,
    encoder_model: EncoderModel,
    encoder_input: EncoderInput,
    species: Species,
    verbose: bool,
) -> np.ndarray:
    """Encode sequences and run the Keras encoder directly, no R/basilisk."""
    from ._aa_props import encode_sequences
    from ._model_cache import load_keras_model

    is_exp = encoder_model.endswith(".EXP")
    x = encode_sequences(sequences, encoder_input, is_exp=is_exp)
    filename = f"{species}_{chain}_{encoder_model}_{encoder_input}_encoder.keras"
    model = load_keras_model(filename)
    return model.predict(x, verbose="auto" if verbose else 0)


def ibex_matrix(
    sequences: Sequence[str | None],
    chain: Chain = "Heavy",
    method: Method = "encoder",
    encoder_model: EncoderModel = "VAE",
    encoder_input: EncoderInput = "atchleyFactors",
    geometric_theta: float = 3.14159265,
    species: Species = "Human",
    verbose: bool = False,
    *,
    fill_value: float = 0.0,
    backend: Backend = "r",
) -> np.ndarray:
    """Embed BCR sequences using Ibex.

    Returns an [N, D] float array aligned to the input sequence list.  ``None``
    entries (missing sequences) receive rows filled with ``fill_value`` (default
    ``0.0``; pass ``float("nan")`` for NaN).  When ``verbose=True`` a warning is
    emitted if any sequences are ``None``.

    Parameters
    ----------
    backend:
        ``"r"`` (default) routes through the Ibex R package + basilisk.
        ``"python"`` loads the Keras model directly in the current Python
        process — requires ``pip install scibex[python-backend]``.
        ``method="geometric"`` always uses the R backend regardless of this
        parameter (geometric is not implemented in the Python backend).
    """
    if method == "geometric":
        backend = "r"

    valid_indices = [i for i, s in enumerate(sequences) if s is not None]
    valid_seqs = [s for s in sequences if s is not None]

    if not valid_seqs:
        raise ValueError("All sequences are None; cannot determine embedding dimension.")

    if backend == "python":
        embedding = _raw_embed_python(
            valid_seqs,
            chain=chain,
            encoder_model=encoder_model,
            encoder_input=encoder_input,
            species=species,
            verbose=verbose,
        )
    else:
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
