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


def _smart_mock(dim: int = 4, val: float = 1.0):
    """Mock for ibex_matrix that respects None → fill_value, matching the real API."""

    def _mock(sequences, fill_value=0.0, **kw):
        out = np.full((len(sequences), dim), fill_value)
        for i, s in enumerate(sequences):
            if s is not None:
                out[i] = val
        return out

    return _mock


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


def test_ibex_matrix_none_fills_zeros_by_default(monkeypatch):
    """ibex_matrix fills None entries with 0.0 by default."""
    import scibex._ibex as mod

    monkeypatch.setattr(mod, "_raw_embed_python", lambda seqs, **kw: np.ones((len(seqs), 4)))
    from scibex import ibex_matrix

    result = ibex_matrix(["CARDYW", None, "CARDSSGYW"])
    assert result.shape == (3, 4)
    assert np.all(result[1] == 0.0)
    assert not np.any(np.isnan(result[[0, 2]]))


def test_ibex_matrix_none_fill_value_nan(monkeypatch):
    """ibex_matrix fills None entries with NaN when fill_value=nan."""
    import scibex._ibex as mod

    monkeypatch.setattr(mod, "_raw_embed_python", lambda seqs, **kw: np.ones((len(seqs), 4)))
    from scibex import ibex_matrix

    result = ibex_matrix(["CARDYW", None], fill_value=float("nan"))
    assert result.shape == (2, 4)
    assert np.all(np.isnan(result[1]))
    assert not np.any(np.isnan(result[0]))


def test_ibex_matrix_verbose_warns_missing(monkeypatch):
    """ibex_matrix emits UserWarning when verbose=True and any sequence is None."""
    import scibex._ibex as mod

    monkeypatch.setattr(mod, "_raw_embed_python", lambda seqs, **kw: np.ones((len(seqs), 4)))
    from scibex import ibex_matrix

    with pytest.warns(UserWarning, match="1 of 2"):
        ibex_matrix(["CARDYW", None], verbose=True)


def test_ibex_matrix_no_warn_when_all_present(monkeypatch):
    """ibex_matrix does not warn when verbose=True and no None entries."""
    import warnings

    import scibex._ibex as mod

    monkeypatch.setattr(mod, "_raw_embed_python", lambda seqs, **kw: np.ones((len(seqs), 4)))
    from scibex import ibex_matrix

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ibex_matrix(["CARDYW", "CARDSSGYW"], verbose=True)


def test_ibex_matrix_returns_full_array_shape(monkeypatch):
    """ibex_matrix always returns [N, D] where N = len(sequences) including None."""
    import scibex._ibex as mod

    monkeypatch.setattr(mod, "_raw_embed_python", lambda seqs, **kw: np.ones((len(seqs), 6)))
    from scibex import ibex_matrix

    result = ibex_matrix(["A", None, "B", None, "C"])
    assert result.shape == (5, 6)


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
    scibex.tl.ibex(adata, chain="Heavy", method="geometric", fill_value=float("nan"))
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


def test_tl_ibex_missing_chain_fills_zeros_by_default(monkeypatch):
    """Missing-chain rows are zero-filled by default, not NaN."""
    import scibex._ibex as _ibex_mod
    import scibex.tl

    monkeypatch.setattr(_ibex_mod, "ibex_matrix", _smart_mock(dim=4))

    adata = _make_airr_adata(_HEAVY_CDRS[:2], extra_light=True)
    scibex.tl.ibex(adata, chain="Heavy", method="geometric")

    assert np.all(adata.obsm["X_ibex"][-1] == 0.0)
    assert not np.any(np.isnan(adata.obsm["X_ibex"][:-1]))


def test_tl_ibex_verbose_warns_missing_chain(monkeypatch):
    """verbose=True emits a UserWarning when chain is missing."""
    import scibex._ibex as _ibex_mod
    import scibex.tl

    monkeypatch.setattr(_ibex_mod, "ibex_matrix", _smart_mock(dim=4))

    adata = _make_airr_adata(_HEAVY_CDRS[:2], extra_light=True)
    with pytest.warns(UserWarning, match="1 of 3 cells missing Heavy chain"):
        scibex.tl.ibex(adata, chain="Heavy", method="geometric", verbose=True)


# -- EXP models (CDR1+CDR2+CDR3) ---------------------------------------------


def _make_airr_adata_exp(heavy_cdr1s, heavy_cdr2s, heavy_cdr3s, extra_no_cdr1=False, extra_no_cdr3=False):
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

    if extra_no_cdr3:
        cell = AirrCell(cell_id="cell_no_cdr3")
        vdj = AirrCell.empty_chain_dict()
        vdj.update({"locus": "IGH", "cdr1_aa": "GFTFSSYA", "cdr2_aa": "ISGSGGST", "productive": True})
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

    def fake_ibex_matrix(sequences, fill_value=0.0, **kwargs):
        captured["sequences"] = list(sequences)
        return np.zeros((len(sequences), 8))

    monkeypatch.setattr(_ibex_mod, "ibex_matrix", fake_ibex_matrix)

    adata = _make_airr_adata_exp(_HEAVY_CDR1S, _HEAVY_CDR2S, _HEAVY_CDRS)
    scibex.tl.ibex(adata, chain="Heavy", method="encoder", encoder_model="CNN.EXP")

    expected = [f"{c1}-{c2}-{c3}" for c1, c2, c3 in zip(_HEAVY_CDR1S, _HEAVY_CDR2S, _HEAVY_CDRS, strict=True)]
    assert captured["sequences"] == expected


def test_tl_ibex_exp_nan_for_missing_cdr1(monkeypatch):
    """Cells missing CDR1 get NaN rows in EXP strict mode when fill_value=nan."""
    import scibex._ibex as _ibex_mod
    import scibex.tl

    monkeypatch.setattr(_ibex_mod, "ibex_matrix", _smart_mock(dim=8))

    adata = _make_airr_adata_exp(_HEAVY_CDR1S[:2], _HEAVY_CDR2S[:2], _HEAVY_CDRS[:2], extra_no_cdr1=True)
    scibex.tl.ibex(
        adata, chain="Heavy", method="encoder", encoder_model="VAE.EXP", strategy="strict", fill_value=float("nan")
    )

    assert adata.obsm["X_ibex"].shape[0] == 3
    assert np.all(np.isnan(adata.obsm["X_ibex"][-1]))  # cell without cdr1_aa
    assert not np.any(np.isnan(adata.obsm["X_ibex"][:-1]))


def test_tl_ibex_exp_missing_cdr1_fills_zeros_by_default(monkeypatch):
    """EXP cells missing CDR1 are zero-filled when strategy='strict'."""
    import scibex._ibex as _ibex_mod
    import scibex.tl

    monkeypatch.setattr(_ibex_mod, "ibex_matrix", _smart_mock(dim=8))

    adata = _make_airr_adata_exp(_HEAVY_CDR1S[:2], _HEAVY_CDR2S[:2], _HEAVY_CDRS[:2], extra_no_cdr1=True)
    scibex.tl.ibex(adata, chain="Heavy", method="encoder", encoder_model="VAE.EXP", strategy="strict")

    assert np.all(adata.obsm["X_ibex"][-1] == 0.0)
    assert not np.any(np.isnan(adata.obsm["X_ibex"][:-1]))


def test_tl_ibex_exp_verbose_warns_missing_cdr1(monkeypatch):
    """verbose=True emits a UserWarning when CDR1/2/3 incomplete in EXP strict mode."""
    import scibex._ibex as _ibex_mod
    import scibex.tl

    monkeypatch.setattr(_ibex_mod, "ibex_matrix", _smart_mock(dim=8))

    adata = _make_airr_adata_exp(_HEAVY_CDR1S[:2], _HEAVY_CDR2S[:2], _HEAVY_CDRS[:2], extra_no_cdr1=True)
    with pytest.warns(UserWarning, match="1 of 3 cells missing CDR"):
        scibex.tl.ibex(adata, chain="Heavy", method="encoder", encoder_model="VAE.EXP", strategy="strict", verbose=True)


# -- strategy param (EXP models) ----------------------------------------------


def test_tl_ibex_exp_strict_passes_none_for_partial_missing(monkeypatch):
    """strategy='strict': cell missing CDR1 → None passed to ibex_matrix."""
    import scibex._ibex as _ibex_mod
    import scibex.tl

    captured = {}

    def fake_ibex_matrix(sequences, fill_value=0.0, **kw):
        captured["sequences"] = list(sequences)
        return np.zeros((len(sequences), 8))

    monkeypatch.setattr(_ibex_mod, "ibex_matrix", fake_ibex_matrix)

    adata = _make_airr_adata_exp(_HEAVY_CDR1S[:2], _HEAVY_CDR2S[:2], _HEAVY_CDRS[:2], extra_no_cdr1=True)
    scibex.tl.ibex(adata, chain="Heavy", method="encoder", encoder_model="CNN.EXP", strategy="strict")

    # last cell has no cdr1_aa/cdr2_aa → strict → None
    assert captured["sequences"][-1] is None
    # first two cells are complete
    assert all(s is not None for s in captured["sequences"][:-1])


def test_tl_ibex_exp_lenient_passes_joined_for_partial_missing(monkeypatch):
    """strategy='lenient': cell missing CDR1 → 'NA-CDR2-CDR3' (not None)."""
    import scibex._ibex as _ibex_mod
    import scibex.tl

    captured = {}

    def fake_ibex_matrix(sequences, fill_value=0.0, **kw):
        captured["sequences"] = list(sequences)
        return np.zeros((len(sequences), 8))

    monkeypatch.setattr(_ibex_mod, "ibex_matrix", fake_ibex_matrix)

    adata = _make_airr_adata_exp(_HEAVY_CDR1S[:2], _HEAVY_CDR2S[:2], _HEAVY_CDRS[:2], extra_no_cdr1=True)
    scibex.tl.ibex(adata, chain="Heavy", method="encoder", encoder_model="CNN.EXP", strategy="lenient")

    # last cell has no cdr1_aa/cdr2_aa → lenient → "NA-NA-CARDYW"
    assert captured["sequences"][-1] is not None
    assert captured["sequences"][-1].startswith("NA-")
    assert captured["sequences"][-1].endswith("-CARDYW")


# -- _join_exp_cdr unit tests -------------------------------------------------


def test_join_exp_cdr_all_present():
    from scibex.tl._ibex import _join_exp_cdr

    assert _join_exp_cdr("A", "B", "C", "strict") == "A-B-C"
    assert _join_exp_cdr("A", "B", "C", "lenient") == "A-B-C"


def test_join_exp_cdr_cdr3_missing_returns_none_for_both_strategies():
    from scibex.tl._ibex import _join_exp_cdr

    nan = float("nan")
    assert _join_exp_cdr("A", "B", nan, "strict") is None
    assert _join_exp_cdr("A", "B", nan, "lenient") is None


def test_join_exp_cdr_cdr1_missing_strict_returns_none():
    from scibex.tl._ibex import _join_exp_cdr

    assert _join_exp_cdr(float("nan"), "B", "C", "strict") is None


def test_join_exp_cdr_cdr1_missing_lenient_substitutes_na():
    from scibex.tl._ibex import _join_exp_cdr

    assert _join_exp_cdr(float("nan"), "B", "C", "lenient") == "NA-B-C"


def test_join_exp_cdr_cdr2_missing_strict_returns_none():
    from scibex.tl._ibex import _join_exp_cdr

    assert _join_exp_cdr("A", float("nan"), "C", "strict") is None


def test_join_exp_cdr_cdr2_missing_lenient_substitutes_na():
    from scibex.tl._ibex import _join_exp_cdr

    assert _join_exp_cdr("A", float("nan"), "C", "lenient") == "A-NA-C"


def test_join_exp_cdr_both_cdrs_missing_strict_returns_none():
    from scibex.tl._ibex import _join_exp_cdr

    nan = float("nan")
    assert _join_exp_cdr(nan, nan, "C", "strict") is None


def test_join_exp_cdr_both_cdrs_missing_lenient_substitutes_both():
    from scibex.tl._ibex import _join_exp_cdr

    nan = float("nan")
    assert _join_exp_cdr(nan, nan, "C", "lenient") == "NA-NA-C"


# -- ibex_matrix edge cases ---------------------------------------------------


def test_ibex_matrix_all_none_raises(monkeypatch):
    """ibex_matrix([None, None]) raises ValueError."""
    import scibex._ibex as mod

    monkeypatch.setattr(mod, "_raw_embed_python", lambda seqs, **kw: np.ones((len(seqs), 4)))
    from scibex import ibex_matrix

    with pytest.raises(ValueError, match="All sequences are None"):
        ibex_matrix([None, None])


def test_ibex_matrix_empty_raises(monkeypatch):
    """ibex_matrix([]) raises ValueError."""
    import scibex._ibex as mod

    monkeypatch.setattr(mod, "_raw_embed_python", lambda seqs, **kw: np.ones((len(seqs), 4)))
    from scibex import ibex_matrix

    with pytest.raises(ValueError):
        ibex_matrix([])


def test_ibex_matrix_single_sequence(monkeypatch):
    """ibex_matrix with one sequence returns shape (1, D)."""
    import scibex._ibex as mod

    monkeypatch.setattr(mod, "_raw_embed_python", lambda seqs, **kw: np.ones((len(seqs), 5)))
    from scibex import ibex_matrix

    result = ibex_matrix(["CARDYW"])
    assert result.shape == (1, 5)


def test_ibex_matrix_verbose_false_no_warn_with_nones(monkeypatch):
    """No warning emitted when verbose=False (default), even with None entries."""
    import warnings

    import scibex._ibex as mod

    monkeypatch.setattr(mod, "_raw_embed_python", lambda seqs, **kw: np.ones((len(seqs), 4)))
    from scibex import ibex_matrix

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ibex_matrix(["CARDYW", None, "CARDSSGYW"])


# -- tl.ibex additional edge cases --------------------------------------------


def test_tl_ibex_light_chain(monkeypatch):
    """chain='Light' extracts light-chain CDR3s and stores obsm correctly."""
    import scirpy as ir
    from scirpy.io import AirrCell, from_airr_cells

    import scibex._ibex as _ibex_mod
    import scibex.tl

    monkeypatch.setattr(_ibex_mod, "ibex_matrix", _smart_mock(dim=4))

    cell = AirrCell(cell_id="cell_0")
    vj = AirrCell.empty_chain_dict()
    vj.update({"locus": "IGK", "junction_aa": _LIGHT_CDRS[0], "productive": True})
    cell.add_chain(vj)
    adata = from_airr_cells([cell])
    ir.pp.index_chains(adata)

    scibex.tl.ibex(adata, chain="Light", method="geometric")

    assert "X_ibex" in adata.obsm
    assert adata.obsm["X_ibex"].shape == (1, 4)
    assert not np.any(np.isnan(adata.obsm["X_ibex"]))


def test_tl_ibex_exp_cdr3_missing_both_strategies(monkeypatch):
    """CDR3=NaN → None passed to ibex_matrix for both strict and lenient."""
    import scibex._ibex as _ibex_mod
    import scibex.tl

    captured = {}

    def fake_ibex_matrix(sequences, fill_value=0.0, **kw):
        captured["sequences"] = list(sequences)
        return np.zeros((len(sequences), 8))

    adata = _make_airr_adata_exp(_HEAVY_CDR1S[:2], _HEAVY_CDR2S[:2], _HEAVY_CDRS[:2], extra_no_cdr3=True)

    for strat in ("strict", "lenient"):
        monkeypatch.setattr(_ibex_mod, "ibex_matrix", fake_ibex_matrix)
        scibex.tl.ibex(adata, chain="Heavy", method="encoder", encoder_model="CNN.EXP", strategy=strat)
        assert captured["sequences"][-1] is None, f"strategy={strat!r}: expected None for missing CDR3"


def test_tl_ibex_exp_cdr2_missing_lenient(monkeypatch):
    """CDR2 missing, lenient → 'CDR1-NA-CDR3' passed to ibex_matrix."""
    import scirpy as ir
    from scirpy.io import AirrCell, from_airr_cells

    import scibex._ibex as _ibex_mod
    import scibex.tl

    captured = {}

    def fake_ibex_matrix(sequences, fill_value=0.0, **kw):
        captured["sequences"] = list(sequences)
        return np.zeros((len(sequences), 8))

    monkeypatch.setattr(_ibex_mod, "ibex_matrix", fake_ibex_matrix)

    # Two-cell adata: normal cell + cell missing CDR2 (so the cdr2_aa column exists but is NaN)
    cells = []
    normal = AirrCell(cell_id="cell_0")
    vdj = AirrCell.empty_chain_dict()
    vdj.update(
        {
            "locus": "IGH",
            "junction_aa": _HEAVY_CDRS[0],
            "cdr1_aa": _HEAVY_CDR1S[0],
            "cdr2_aa": _HEAVY_CDR2S[0],
            "productive": True,
        }
    )
    normal.add_chain(vdj)
    cells.append(normal)
    no_cdr2 = AirrCell(cell_id="cell_no_cdr2")
    vdj2 = AirrCell.empty_chain_dict()
    vdj2.update({"locus": "IGH", "junction_aa": "CARDYW", "cdr1_aa": "GFTFSSYA", "productive": True})
    no_cdr2.add_chain(vdj2)
    cells.append(no_cdr2)
    adata = from_airr_cells(cells)
    ir.pp.index_chains(adata)

    scibex.tl.ibex(adata, chain="Heavy", method="encoder", encoder_model="CNN.EXP", strategy="lenient")
    assert captured["sequences"][-1] == "GFTFSSYA-NA-CARDYW"


def test_tl_ibex_exp_both_cdrs_missing_lenient(monkeypatch):
    """CDR1+CDR2 missing, lenient → 'NA-NA-CDR3' passed to ibex_matrix."""
    import scibex._ibex as _ibex_mod
    import scibex.tl

    captured = {}

    def fake_ibex_matrix(sequences, fill_value=0.0, **kw):
        captured["sequences"] = list(sequences)
        return np.zeros((len(sequences), 8))

    monkeypatch.setattr(_ibex_mod, "ibex_matrix", fake_ibex_matrix)

    # extra_no_cdr1 cell has junction_aa="CARDYW" but no cdr1_aa/cdr2_aa
    adata = _make_airr_adata_exp(_HEAVY_CDR1S[:1], _HEAVY_CDR2S[:1], _HEAVY_CDRS[:1], extra_no_cdr1=True)
    scibex.tl.ibex(adata, chain="Heavy", method="encoder", encoder_model="CNN.EXP", strategy="lenient")

    assert captured["sequences"][-1] == "NA-NA-CARDYW"


def test_tl_ibex_geometric_ignores_strategy(monkeypatch):
    """strategy has no effect for non-EXP models; both produce identical sequence lists."""
    import scibex._ibex as _ibex_mod
    import scibex.tl

    results = {}

    def make_mock(key):
        def fake_ibex_matrix(sequences, fill_value=0.0, **kw):
            results[key] = list(sequences)
            return np.zeros((len(sequences), 4))

        return fake_ibex_matrix

    adata = _make_airr_adata(_HEAVY_CDRS)

    monkeypatch.setattr(_ibex_mod, "ibex_matrix", make_mock("strict"))
    scibex.tl.ibex(adata, chain="Heavy", method="geometric", strategy="strict")

    monkeypatch.setattr(_ibex_mod, "ibex_matrix", make_mock("lenient"))
    scibex.tl.ibex(adata, chain="Heavy", method="geometric", strategy="lenient")

    assert results["strict"] == results["lenient"]
    assert all(s is not None for s in results["strict"])
