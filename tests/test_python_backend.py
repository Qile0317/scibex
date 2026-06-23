"""Tests for the Python backend (backend="python") of ibex_matrix and tl.ibex.

These tests require both:
  - immApex via rpy2 (require_ibex_r)  — for encoding lookup tables
  - keras in the current Python env (require_keras_python) — for model loading

Tests skip gracefully when either dependency is missing.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# _model_cache
# ---------------------------------------------------------------------------


def test_model_cache_dir_returns_path():
    from pathlib import Path

    from scibex._model_cache import model_cache_dir

    d = model_cache_dir()
    assert isinstance(d, Path)


def test_model_cache_dir_matches_r_user_dir(require_ibex_r):
    """Python cache dir must match tools::R_user_dir('Ibex', 'cache') so models
    downloaded by the R package are reused by the Python backend."""
    from pathlib import Path

    import rpy2.robjects as ro

    from scibex._model_cache import model_cache_dir

    r_path = ro.r("tools::R_user_dir('Ibex', 'cache')")[0]  # type: ignore
    assert model_cache_dir() == Path(r_path)


def test_get_model_path_returns_existing_file(require_keras_python, require_ibex_r):
    """get_model_path for a model already in the R cache returns an existing file."""
    from scibex._model_cache import get_model_path

    fname = "Human_Heavy_VAE_atchleyFactors_encoder.keras"
    p = get_model_path(fname)
    assert p.exists(), f"Expected model at {p}"


def test_load_keras_model_returns_model(require_keras_python, require_ibex_r):
    """load_keras_model loads the .keras file and returns a callable keras Model."""
    import keras

    from scibex._model_cache import load_keras_model

    model = load_keras_model("Human_Heavy_VAE_atchleyFactors_encoder.keras")
    assert isinstance(model, keras.Model)


def test_load_keras_model_is_cached(require_keras_python, require_ibex_r):
    """load_keras_model returns the same object on repeated calls (cached)."""
    from scibex._model_cache import load_keras_model

    fname = "Human_Heavy_CNN_atchleyFactors_encoder.keras"
    m1 = load_keras_model(fname)
    m2 = load_keras_model(fname)
    assert m1 is m2


# ---------------------------------------------------------------------------
# ibex_matrix — backend="python" param
# ---------------------------------------------------------------------------


def test_ibex_matrix_accepts_backend_r():
    """backend='r' (default) is accepted without error even without keras."""
    import inspect

    import scibex._ibex as mod

    sig = inspect.signature(mod.ibex_matrix)
    assert "backend" in sig.parameters


def test_ibex_matrix_accepts_backend_python():
    import inspect

    import scibex._ibex as mod

    sig = inspect.signature(mod.ibex_matrix)
    assert "backend" in sig.parameters


def test_ibex_matrix_python_backend_shape(require_keras_python, require_ibex_r):
    from scibex import ibex_matrix

    seqs = ["CARDYW", "CARDSSGYW", "CARDTGYW"]
    emb = ibex_matrix(seqs, chain="Heavy", encoder_model="VAE", encoder_input="atchleyFactors", backend="python")
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 2
    assert emb.shape[0] == len(seqs)


def test_ibex_matrix_python_backend_returns_float(require_keras_python, require_ibex_r):
    from scibex import ibex_matrix

    emb = ibex_matrix(["CARDYW"], backend="python")
    assert np.issubdtype(emb.dtype, np.floating)


def test_ibex_matrix_python_backend_none_fill(require_keras_python, require_ibex_r):
    """None entries are filled with fill_value (same as R backend)."""
    from scibex import ibex_matrix

    result = ibex_matrix(["CARDYW", None, "CARDSSGYW"], backend="python")
    assert result.shape[0] == 3
    assert np.all(result[1] == 0.0)


def test_ibex_matrix_python_backend_parity_with_r(require_keras_python, require_ibex_r):
    """Python backend output must be close to R backend output."""
    from scibex import ibex_matrix

    seqs = ["CARDYW", "CARDSSGYW", "CARDTGYW"]
    emb_r = ibex_matrix(seqs, chain="Heavy", encoder_model="CNN", encoder_input="atchleyFactors", backend="r")
    emb_py = ibex_matrix(seqs, chain="Heavy", encoder_model="CNN", encoder_input="atchleyFactors", backend="python")

    assert emb_r.shape == emb_py.shape
    np.testing.assert_allclose(emb_r, emb_py, atol=1e-4)


def test_ibex_matrix_python_backend_geometric_falls_back_to_r(require_keras_python, require_ibex_r):
    """geometric method auto-falls back to R backend even when backend='python'."""
    from scibex import ibex_matrix

    emb = ibex_matrix(["CARDYW"], method="geometric", backend="python")
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 2


# ---------------------------------------------------------------------------
# tl.ibex — backend="python" param
# ---------------------------------------------------------------------------


def test_tl_ibex_accepts_backend_param():
    import inspect

    import scibex.tl._ibex as mod

    sig = inspect.signature(mod.ibex)
    assert "backend" in sig.parameters


def test_tl_ibex_python_backend_stores_obsm(require_keras_python, require_ibex_r):
    import scirpy as ir
    from scirpy.io import AirrCell, from_airr_cells

    import scibex.tl

    cell = AirrCell(cell_id="c0")
    vdj = AirrCell.empty_chain_dict()
    vdj.update({"locus": "IGH", "junction_aa": "CARDYW", "productive": True})
    cell.add_chain(vdj)
    adata = from_airr_cells([cell])
    ir.pp.index_chains(adata)

    scibex.tl.ibex(adata, chain="Heavy", encoder_model="VAE", encoder_input="atchleyFactors", backend="python")
    assert "X_ibex" in adata.obsm
    assert adata.obsm["X_ibex"].shape[0] == 1
