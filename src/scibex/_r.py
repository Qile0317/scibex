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


def setup(lib_loc: str | None = None) -> None:
    """Prepend a custom R library directory to R's ``.libPaths()``.

    Call once before any ``ibex_matrix`` / ``tl.ibex`` call.  The path is
    prepended to R's search path so every subsequent ``importr()`` call —
    Ibex, immApex, or any other R package — finds packages there
    automatically.  If called after packages have already been loaded, the
    Ibex singleton is cleared and reloaded from the updated path on the next
    call.

    Parameters
    ----------
    lib_loc:
        Path to prepend to R's ``.libPaths()`` (e.g. an renv library).
        ``None`` is a no-op (useful for resetting the singleton in tests).

    Examples
    --------
    Use Ibex from an renv-managed library:

    >>> import rpy2.robjects as ro
    >>> import scibex as ib
    >>> renv_lib = ro.r("renv::paths$library()")[0]
    >>> ib.setup(lib_loc=renv_lib)
    """
    global _ibex_pkg
    with _lock:
        if lib_loc is not None:
            import rpy2.robjects as ro

            ro.r(f'.libPaths(c("{lib_loc}", .libPaths()))')
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
            # rpy2 pre-initialises Python at the C level, but reticulate's
            # py_available() returns FALSE until reticulate itself calls
            # py_config().  Without this call, basiliskRun uses inline mode
            # (thinking Python isn't running), calls Py_Initialize for IbexEnv,
            # which is a no-op because Python 3.x is already live — leaving
            # reticulate attached to the wrong interpreter.  Calling py_config()
            # first makes py_available() return TRUE, so basilisk routes to
            # PSOCK subprocess mode where IbexEnv's Python starts cleanly.
            import rpy2.robjects as ro

            ro.r("reticulate::py_config()")
            _ibex_pkg = importr("Ibex")
    return _ibex_pkg
