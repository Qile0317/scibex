import atexit
import os

import pytest

# When inside a conda env, point rpy2 at conda's R before any import touches it.
if _conda_prefix := os.environ.get("CONDA_PREFIX"):
    _conda_r_home = f"{_conda_prefix}/lib/R"
    if os.path.isdir(_conda_r_home):
        os.environ.setdefault("R_HOME", _conda_r_home)

# Prevent ancestor-directory .Rprofile files (e.g. a sibling project's renv)
# from being sourced when rpy2 initialises the embedded R runtime.
os.environ.setdefault("R_PROFILE_USER", "/dev/null")

# Allow callers to inject an R library path (e.g. an renv library) via env var
# before R is initialised.  Example:
#   SCIBEX_R_LIB=/path/to/renv/library/linux-ubuntu-noble/R-4.5/x86_64 just test
if _lib := os.environ.get("SCIBEX_R_LIB"):
    _existing = os.environ.get("R_LIBS_USER", "")
    os.environ["R_LIBS_USER"] = f"{_lib}:{_existing}" if _existing else _lib


@pytest.fixture(scope="session", autouse=True)
def _suppress_rpy2_endr():
    """Prevent the rpy2 shutdown crash on Python 3.13.

    rpy2 registers embedded.endr() via atexit to tear down the embedded R
    runtime.  On Python 3.13, _PyThreadState_Attach gained a stricter check
    that raises a fatal error when endr() runs during interpreter shutdown
    (rpy2 issues #872/#1178).  Unregistering endr() after all tests have run
    avoids the crash: the OS reclaims R's memory on process exit anyway.
    """
    yield
    try:
        from rpy2.rinterface_lib import embedded

        atexit.unregister(embedded.endr)
    except Exception:
        pass
