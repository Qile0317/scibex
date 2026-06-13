from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData
    from muon import MuData

from .._types import Chain, EncoderInput, EncoderModel, Method, Species, Strategy


def _join_exp_cdr(c1: Any, c2: Any, c3: Any, strategy: Strategy) -> str | None:
    """Join CDR1/CDR2/CDR3 into an EXP model input string.

    Returns None when the cell should be filled (strict: any CDR missing;
    lenient: CDR3 missing or all missing).  For lenient with partial CDRs,
    substitutes missing slots with the literal "NA" to match Ibex's R convention
    (paste(NA, cdr2, cdr3, sep="-") → "NA-CDR2-CDR3").
    """
    if pd.isna(c3):
        return None
    if strategy == "lenient":
        r1 = "NA" if pd.isna(c1) else c1
        r2 = "NA" if pd.isna(c2) else c2
        return f"{r1}-{r2}-{c3}"
    # strict: any missing CDR → fill row
    if pd.isna(c1) or pd.isna(c2):
        return None
    return f"{c1}-{c2}-{c3}"


def ibex(
    adata: AnnData | MuData,
    chain: Chain = "Heavy",
    method: Method = "encoder",
    encoder_model: EncoderModel = "VAE",
    encoder_input: EncoderInput = "atchleyFactors",
    geometric_theta: float = 3.14159265,
    species: Species = "Human",
    verbose: bool = False,
    *,
    fill_value: float = 0.0,
    strategy: Strategy = "lenient",
    airr_mod: str = "airr",
    airr_key: str = "airr",
    chain_idx_key: str = "chain_indices",
    cdr1_key: str = "cdr1_aa",
    cdr2_key: str = "cdr2_aa",
    key_added: str = "X_ibex",
) -> None:
    """Embed BCR sequences from a scirpy AnnData/MuData using Ibex.

    Reads sequences from obsm["chain_indices"] (requires ir.pp.index_chains to
    have been called) and stores embeddings in adata.obsm[key_added].  Cells
    lacking the requested chain or CDR data receive rows filled with
    ``fill_value`` (default ``0.0``; pass ``float("nan")`` for NaN).  When
    ``verbose=True`` a warning is emitted with the count of affected cells.

    For EXP encoder models (CNN.EXP, VAE.EXP), CDR1 and CDR2 are extracted
    alongside CDR3 and joined as "CDR1-CDR2-CDR3".  ``strategy`` controls how
    partial missingness is handled:

    - ``"lenient"`` (default): missing CDR3 → fill row; missing CDR1/CDR2 only →
      substitute with the literal ``"NA"`` (Ibex's R convention) and embed.
    - ``"strict"``: any missing CDR → fill row with ``fill_value``.

    For non-EXP models, ``strategy`` has no effect (CDR3 presence is binary).
    """
    import scirpy as ir
    from muon import MuData as _MuData

    from .._ibex import ibex_matrix

    _adata = adata.mod[airr_mod] if isinstance(adata, _MuData) else adata
    chain_pos = "VDJ_1" if chain == "Heavy" else "VJ_1"
    is_exp = encoder_model in ("CNN.EXP", "VAE.EXP")

    if is_exp:
        cdr1 = ir.get.airr(_adata, cdr1_key, chain_pos)
        cdr2 = ir.get.airr(_adata, cdr2_key, chain_pos)
        cdr3 = ir.get.airr(_adata, "junction_aa", chain_pos)
        seqs: list[str | None] = [
            _join_exp_cdr(c1, c2, c3, strategy) for c1, c2, c3 in zip(cdr1, cdr2, cdr3, strict=False)
        ]
    else:
        cdr3 = ir.get.airr(_adata, "junction_aa", chain_pos)
        seqs = [None if pd.isna(s) else s for s in cdr3]

    if verbose:
        n_missing = sum(1 for s in seqs if s is None)
        if n_missing:
            if is_exp:
                warnings.warn(
                    f"scibex: {n_missing} of {_adata.n_obs} cells missing CDR1/CDR2/CDR3 "
                    f"for {chain} EXP model; rows filled with {fill_value}.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"scibex: {n_missing} of {_adata.n_obs} cells missing {chain} chain; "
                    f"rows filled with {fill_value}.",
                    UserWarning,
                    stacklevel=2,
                )

    embedding = ibex_matrix(
        seqs,
        chain=chain,
        method=method,
        encoder_model=encoder_model,
        encoder_input=encoder_input,
        geometric_theta=geometric_theta,
        species=species,
        verbose=False,  # tl.ibex emits contextual warning above; avoid double-warn
        fill_value=fill_value,
    )
    _adata.obsm[key_added] = embedding
