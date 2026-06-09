from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData
    from muon import MuData


def ibex(
    adata: AnnData | MuData,
    chain: Literal["Heavy", "Light"] = "Heavy",
    method: Literal["encoder", "geometric"] = "geometric",
    encoder_model: Literal["CNN", "VAE", "CNN.EXP", "VAE.EXP"] = "VAE",
    encoder_input: str = "atchleyFactors",
    geometric_theta: float = 3.14159265,
    species: str = "Human",
    verbose: bool = False,
    *,
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
    lacking the requested chain receive all-NaN rows.

    For EXP encoder models (CNN.EXP, VAE.EXP), CDR1 and CDR2 are extracted
    alongside CDR3 and joined as "CDR1-CDR2-CDR3".  cdr1_key/cdr2_key name the
    AIRR fields to read (default "cdr1_aa"/"cdr2_aa" per AIRR schema v1.4).
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
        valid: pd.Series = pd.notna(cdr1) & pd.notna(cdr2) & pd.notna(cdr3)
        valid_seqs = [f"{c1}-{c2}-{c3}" for c1, c2, c3 in zip(cdr1[valid], cdr2[valid], cdr3[valid], strict=True)]
    else:
        cdr3 = ir.get.airr(_adata, "junction_aa", chain_pos)
        valid = pd.notna(cdr3)
        valid_seqs = cdr3[valid].tolist()

    embedding = ibex_matrix(
        valid_seqs,
        chain=chain,
        method=method,
        encoder_model=encoder_model,
        encoder_input=encoder_input,
        geometric_theta=geometric_theta,
        species=species,
        verbose=verbose,
    )

    full = np.full((_adata.n_obs, embedding.shape[1]), np.nan)
    full[valid.values] = embedding
    _adata.obsm[key_added] = full
