"""Tests for Ibex R wrapper (ibex_matrix and tl.ibex)."""

import numpy as np
import pytest

pytest.importorskip("rpy2", reason="rpy2 not installed")


# -- helpers ------------------------------------------------------------------

_HEAVY_CDRS = ["CARDYW", "CARDSSGYW", "CARDTGYW"]
_LIGHT_CDRS = ["CQQYGTSF", "CQQSYSTPF"]

# Typical IGH CDR1/CDR2 sequences for testing EXP models
_HEAVY_CDR1S = ["GFTFSSYA", "GGTFSSYA", "GFTFSNYA"]
_HEAVY_CDR2S = ["ISGSGGST", "ISGSAGST", "ISGSGGNT"]


@pytest.fixture(scope="session")
def require_ibex_r():
    from rpy2.robjects.packages import isinstalled

    if not isinstalled("Ibex"):
        pytest.skip("Ibex R package not installed")


@pytest.fixture(scope="session")
def require_keras(require_ibex_r):
    """Skip if the Ibex basilisk environment cannot import keras.

    The encoder uses basiliskRun(env = IbexEnv, ...) which manages its own
    Python environment — plain reticulate::import() would test the wrong thing.
    """
    import rpy2.robjects as ro

    try:
        ro.r('basilisk::basiliskRun(env = Ibex:::IbexEnv, fun = function() reticulate::import("keras"))')
    except Exception:
        pytest.skip("keras not accessible via Ibex basilisk IbexEnv")


def _make_airr_adata(heavy_cdrs, extra_light=False):
    """Build a minimal scirpy AIRR AnnData from a list of heavy-chain CDR3s.

    If extra_light=True, appends one extra cell that has only a light chain
    (no heavy chain), to test the NaN-fill path.
    """
    import scirpy as ir
    from scirpy.io import AirrCell, from_airr_cells

    cells = []
    for i, cdr3 in enumerate(heavy_cdrs):
        cell = AirrCell(cell_id=f"cell_{i}")
        vdj = AirrCell.empty_chain_dict()
        vdj.update({"locus": "IGH", "junction_aa": cdr3, "productive": True})
        cell.add_chain(vdj)
        cells.append(cell)

    if extra_light:
        cell = AirrCell(cell_id="cell_light_only")
        vj = AirrCell.empty_chain_dict()
        vj.update({"locus": "IGK", "junction_aa": "CQQYGTSF", "productive": True})
        cell.add_chain(vj)
        cells.append(cell)

    adata = from_airr_cells(cells)
    ir.pp.index_chains(adata)
    return adata


# -- _r.py --------------------------------------------------------------------


def test_ibex_r_returns_package(require_ibex_r):
    from scibex._r import ibex_r

    pkg = ibex_r()
    assert pkg is not None


def test_ibex_r_is_cached(require_ibex_r):
    from scibex._r import ibex_r

    assert ibex_r() is ibex_r()


# -- ibex_matrix --------------------------------------------------------------


def test_ibex_matrix_heavy_chain_shape(require_ibex_r):
    from scibex import ibex_matrix

    emb = ibex_matrix(_HEAVY_CDRS, chain="Heavy", method="geometric")
    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] == len(_HEAVY_CDRS)
    assert emb.ndim == 2


def test_ibex_matrix_light_chain_shape(require_ibex_r):
    from scibex import ibex_matrix

    emb = ibex_matrix(_LIGHT_CDRS, chain="Light", method="geometric")
    assert emb.shape[0] == len(_LIGHT_CDRS)
    assert emb.ndim == 2


def test_ibex_matrix_consistent_dim_across_chains(require_ibex_r):
    """Heavy and light embeddings should have the same number of dimensions."""
    from scibex import ibex_matrix

    heavy = ibex_matrix(_HEAVY_CDRS, chain="Heavy", method="geometric")
    light = ibex_matrix(_LIGHT_CDRS, chain="Light", method="geometric")
    assert heavy.shape[1] == light.shape[1]


def test_ibex_matrix_returns_float(require_ibex_r):
    from scibex import ibex_matrix

    emb = ibex_matrix(_HEAVY_CDRS[:1], chain="Heavy", method="geometric")
    assert np.issubdtype(emb.dtype, np.floating)


# -- tl.ibex ------------------------------------------------------------------


def test_tl_ibex_stores_obsm(require_ibex_r):
    import scibex.tl

    adata = _make_airr_adata(_HEAVY_CDRS)
    scibex.tl.ibex(adata, chain="Heavy", method="geometric")
    assert "X_ibex" in adata.obsm


def test_tl_ibex_obsm_aligned_to_obs(require_ibex_r):
    import scibex.tl

    adata = _make_airr_adata(_HEAVY_CDRS)
    scibex.tl.ibex(adata, chain="Heavy", method="geometric")
    assert adata.obsm["X_ibex"].shape[0] == adata.n_obs


def test_tl_ibex_nan_for_missing_chain(require_ibex_r):
    import scibex.tl

    adata = _make_airr_adata(_HEAVY_CDRS, extra_light=True)
    scibex.tl.ibex(adata, chain="Heavy", method="geometric")
    # last cell has no heavy chain — entire row must be NaN
    assert np.all(np.isnan(adata.obsm["X_ibex"][-1]))
    # cells with chains must not be NaN
    assert not np.any(np.isnan(adata.obsm["X_ibex"][:-1]))


def test_tl_ibex_key_added(require_ibex_r):
    import scibex.tl

    adata = _make_airr_adata(_HEAVY_CDRS[:1])
    scibex.tl.ibex(adata, chain="Heavy", method="geometric", key_added="X_custom")
    assert "X_custom" in adata.obsm
    assert "X_ibex" not in adata.obsm


def test_tl_ibex_mudata_stores_in_airr_modality(require_ibex_r):
    import anndata as ad
    import muon as mu

    import scibex.tl

    adata_airr = _make_airr_adata(_HEAVY_CDRS[:2])
    adata_gex = ad.AnnData(obs=adata_airr.obs.copy())
    mdata = mu.MuData({"gex": adata_gex, "airr": adata_airr})

    scibex.tl.ibex(mdata, chain="Heavy", method="geometric")

    assert "X_ibex" in mdata["airr"].obsm
    assert mdata["airr"].obsm["X_ibex"].shape[0] == 2


# -- EXP models (CDR1+CDR2+CDR3) ---------------------------------------------


def _make_airr_adata_exp(heavy_cdr1s, heavy_cdr2s, heavy_cdr3s, extra_no_cdr1=False):
    """Build AIRR AnnData with cdr1_aa and cdr2_aa fields populated."""
    import scirpy as ir
    from scirpy.io import AirrCell, from_airr_cells

    cells = []
    for i, (c1, c2, c3) in enumerate(zip(heavy_cdr1s, heavy_cdr2s, heavy_cdr3s, strict=True)):
        cell = AirrCell(cell_id=f"cell_{i}")
        vdj = AirrCell.empty_chain_dict()
        vdj.update({"locus": "IGH", "junction_aa": c3, "cdr1_aa": c1, "cdr2_aa": c2, "productive": True})
        cell.add_chain(vdj)
        cells.append(cell)

    if extra_no_cdr1:
        cell = AirrCell(cell_id="cell_no_cdr1")
        vdj = AirrCell.empty_chain_dict()
        vdj.update({"locus": "IGH", "junction_aa": "CARDYW", "productive": True})
        cell.add_chain(vdj)
        cells.append(cell)

    adata = from_airr_cells(cells)
    ir.pp.index_chains(adata)
    return adata


def test_tl_ibex_exp_passes_joined_sequences(monkeypatch):
    """tl.ibex with CNN.EXP should pass CDR1-CDR2-CDR3 strings to ibex_matrix."""
    import scibex._ibex as _ibex_mod
    import scibex.tl

    captured = {}

    def fake_ibex_matrix(sequences, **kwargs):
        captured["sequences"] = sequences
        n = len(sequences)
        return np.zeros((n, 8))

    monkeypatch.setattr(_ibex_mod, "ibex_matrix", fake_ibex_matrix)

    adata = _make_airr_adata_exp(_HEAVY_CDR1S, _HEAVY_CDR2S, _HEAVY_CDRS)
    scibex.tl.ibex(adata, chain="Heavy", method="encoder", encoder_model="CNN.EXP")

    expected = [f"{c1}-{c2}-{c3}" for c1, c2, c3 in zip(_HEAVY_CDR1S, _HEAVY_CDR2S, _HEAVY_CDRS, strict=True)]
    assert captured["sequences"] == expected


def test_tl_ibex_exp_nan_for_missing_cdr1(monkeypatch):
    """Cells missing CDR1 get NaN rows in EXP mode."""
    import scibex._ibex as _ibex_mod
    import scibex.tl

    def fake_ibex_matrix(sequences, **kwargs):
        return np.ones((len(sequences), 8))

    monkeypatch.setattr(_ibex_mod, "ibex_matrix", fake_ibex_matrix)

    adata = _make_airr_adata_exp(_HEAVY_CDR1S[:2], _HEAVY_CDR2S[:2], _HEAVY_CDRS[:2], extra_no_cdr1=True)
    scibex.tl.ibex(adata, chain="Heavy", method="encoder", encoder_model="VAE.EXP")

    assert adata.obsm["X_ibex"].shape[0] == 3
    assert np.all(np.isnan(adata.obsm["X_ibex"][-1]))  # cell without cdr1_aa
    assert not np.any(np.isnan(adata.obsm["X_ibex"][:-1]))
