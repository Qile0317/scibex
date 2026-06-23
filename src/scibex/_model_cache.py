"""Model download and caching for the Python backend.

Uses R's tools::R_user_dir("Ibex", "cache") as the cache directory so models
downloaded by the R package are reused by the Python backend (no double download).
Missing models are fetched from the same Zenodo record as the R package.
"""

from __future__ import annotations

import urllib.request
from functools import cache
from pathlib import Path

_ZENODO_BASE = "https://zenodo.org/record/14919286/files"


def model_cache_dir() -> Path:
    """Return the Ibex model cache directory via R's tools::R_user_dir."""
    import rpy2.robjects as ro

    return Path(ro.r("tools::R_user_dir('Ibex', 'cache')")[0])  # type: ignore


def get_model_path(filename: str) -> Path:
    """Return local path to the model, downloading from Zenodo if not cached."""
    cache_dir = model_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / filename
    if not path.exists():
        url = f"{_ZENODO_BASE}/{filename}"
        urllib.request.urlretrieve(url, path)
    return path


@cache
def load_keras_model(filename: str):
    """Load a .keras model by filename, caching the loaded object in-process."""
    # The shipped .keras models have dotted layer names, which the torch backend
    # rejects (ParameterDict forbids "."); the python-backend extra ships
    # tensorflow, so default Keras to it.  setdefault respects an explicit
    # KERAS_BACKEND (e.g. jax, which also loads these models).
    import os

    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # suppress C++ INFO/WARNING/GPU noise
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # suppress oneDNN startup message
    import keras

    path = get_model_path(filename)
    return keras.saving.load_model(path)
