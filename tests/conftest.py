import os

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
