from __future__ import annotations

import os
import threading

# When inside a conda env, prefer conda's R over any system R that may also
# be on PATH.  Must be set before the first rpy2 import that touches R.
if _conda_prefix := os.environ.get("CONDA_PREFIX"):
    _conda_r_home = f"{_conda_prefix}/lib/R"
    if os.path.isdir(_conda_r_home):
        os.environ.setdefault("R_HOME", _conda_r_home)

from rpy2.robjects.packages import importr

_lock = threading.Lock()
_ibex_pkg = None
_lib_loc: str | None = None


def setup(lib_loc: str | None = None) -> None:
    """Configure the R library location used when loading the Ibex package.

    Call once before any ``ibex_matrix`` / ``tl.ibex`` call. If called after
    the package has already been loaded, the cached instance is cleared and
    the package is reloaded with the new ``lib_loc`` on the next call.

    Parameters
    ----------
    lib_loc:
        Path to the R library directory that contains Ibex (e.g. an renv
        library). ``None`` uses R's default ``.libPaths()``.

    Examples
    --------
    Use Ibex from an renv-managed library:

    >>> import rpy2.robjects as ro
    >>> import scibex as ib
    >>> renv_lib = ro.r("renv::paths$library()")[0]
    >>> ib.setup(lib_loc=renv_lib)
    """
    global _lib_loc, _ibex_pkg
    with _lock:
        _lib_loc = lib_loc
        _ibex_pkg = None


def ibex_r():
    """Return the cached Ibex R package, loading it if necessary.

    Thread-safe: concurrent callers block until the first initialisation
    completes; R's embedded runtime is single-threaded.
    """
    global _ibex_pkg
    if _ibex_pkg is not None:
        return _ibex_pkg
    with _lock:
        if _ibex_pkg is None:
            _ibex_pkg = importr("Ibex", lib_loc=_lib_loc)
    return _ibex_pkg
