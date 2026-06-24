import atexit
import os

import pytest

# Point rpy2 at conda's R before any import touches it.  Library paths are
# handled entirely by the project's renv: rpy2's embedded R sources scibex's
# .Rprofile (the renv autoloader) on startup, which puts the renv library on
# .libPaths() — so Ibex is found without any R_LIBS_USER / SCIBEX_R_LIB munging.
# Run `just setup-r` once to restore the renv from renv.lock.
_conda_prefix = os.environ.get("CONDA_PREFIX", "")
if _conda_prefix:
    _conda_r_home = f"{_conda_prefix}/lib/R"
    if os.path.isdir(_conda_r_home):
        os.environ.setdefault("R_HOME", _conda_r_home)

# The shipped .keras models have dotted layer names that the torch backend
# rejects, so the python-backend extra uses tensorflow.  Set all TF env vars
# before anything imports keras — they must be set before TF dlopen's its
# shared library, which happens at import time.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # suppress C++ INFO/WARNING
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # suppress oneDNN message


@pytest.fixture(scope="session")
def require_keras_python():
    """Skip if the python-backend extra is not installed."""
    from scibex.utils import has_python_backend

    if not has_python_backend():
        pytest.skip("keras not installed (install scibex[python-backend])")


@pytest.fixture(scope="session")
def require_ibex_r():
    """Skip if the Ibex R package is not installed; propagate R init errors as failures."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import isinstalled
    except ImportError:
        pytest.skip("rpy2 not installed")
        return
    if not isinstalled("Ibex"):
        libpaths = list(ro.r(".libPaths()"))  # type: ignore
        pytest.skip(f"Ibex not found in R .libPaths(): {libpaths}")


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
