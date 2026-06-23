# Installation

## Python package

```sh
pip install scibex
# or with uv
uv add scibex
```

### Python backend

scibex loads encoder models directly in Python via Keras and TensorFlow, bypassing
R / rpy2 / basilisk entirely.  These are included as core dependencies — no extra
install step is required.

If you already have JAX set up and prefer it as the Keras backend, install
`keras>=3.6` separately and set `KERAS_BACKEND=jax`.  Note: the PyTorch backend
cannot load the shipped models (dotted layer names; `torch` `ParameterDict` forbids
`.`).

## R dependency

scibex is a Python wrapper around the [Ibex R package](https://github.com/BorchLab/Ibex).
The R package must be installed and visible to the R runtime that rpy2 uses before
calling any `scibex` function.

### Option A — install from Python (recommended)

```python
import scibex as ib

ib.install_r_deps()                           # installs into R's default .libPaths()
ib.install_r_deps(lib_loc="/path/to/my/Rlib") # install into a specific directory
ib.install_r_deps(force=True)                 # force-reinstall everything
```

This installs Ibex, `remotes`, and `callr`.  Packages already present are
skipped, so repeated calls return quickly.  `callr` is required for the
encoder (`method="encoder"`) to work from a Jupyter notebook: without it,
basilisk runs inline and conflicts with rpy2's pre-initialized Python.

If the target directory is non-standard, tell scibex where to find it at runtime:

```python
ib.setup(lib_loc="/path/to/my/Rlib")
```

`setup()` must be called **before** the first `ib.tl.ibex(...)` or `ib.ibex_matrix(...)` call.

### Option B — install directly in R

```r
remotes::install_github("BorchLab/Ibex@devel")
```

## Troubleshooting R environments

### rpy2 / conda: base packages not found at startup

```text
During startup - Warning messages:
1: package 'grDevices' in options("defaultPackages") was not found
2: package 'graphics' in options("defaultPackages") was not found
3: package 'stats' in options("defaultPackages") was not found
```

This means rpy2's C extension was compiled against a different `libR.so` than the
one in your conda environment. It happens when rpy2 is installed from PyPI inside a
conda env — the PyPI wheel is not compiled with the `-rpath` flag pointing at
conda's R library.

**Fix (option A) — recompile from source in the conda env:**

```bash
conda activate ibex   # or your env name
LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib/R/lib" \
    pip install --force-reinstall --no-binary rpy2 rpy2
```

**Fix (option B) — use the conda-forge build (pre-patched):**

```bash
conda install -n ibex -c conda-forge rpy2
```

### rpy2 picks up the wrong R installation

rpy2 uses whichever `R` binary appears first on `PATH`. Check which one it will use:

```bash
R --version        # should match your intended R installation
Rscript -e "R.home()"
```

If you are using conda, activate the correct environment before starting Python.

### `list.files` arity error (R ABI mismatch)

```text
RRuntimeError: Error in list.files(...) :
  8 arguments passed to .Internal(list.files) which requires 9
```

R 4.5+ changed `.Internal(list.files)` from 8 to 9 arguments.  This error means
the C runtime (the `libR.so` loaded by rpy2) is R 4.5+ but the R bytecode being
executed was compiled for R ≤ 4.4.

**Case A — conda R only (partial upgrade).** After `conda update r-base`, the
base package bytecode may lag the C library.  Fix:

```bash
conda install -n <env> -c conda-forge r-base --force-reinstall
```

Then reinstall any R packages that depend on compiled C/C++/Fortran code.

**Case B — conda R 4.5.x + system R 4.6+.** If your machine has both a conda R
(e.g. 4.5.1) and a system R (e.g. 4.6.0), rpy2's compiled CFFI extension may
load the *system* `libR.so` via its rpath, while `R_HOME` points to the conda
install.  The result is an ABI mismatch between the system R 4.6 C code and the
conda R 4.5 bytecode.

Fix — recompile rpy2 and patch the binary's rpath:

```bash
conda activate <env>
# Install patchelf if not already present:
conda install -c conda-forge patchelf
# Recompile rpy2 and patch the CFFI extension in one step:
just setup-r
```

If you are not using `just`, the equivalent manual steps are:

```bash
LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib/R/lib" \
    pip install --force-reinstall --no-binary rpy2 rpy2

CFFI_SO=$(find $VIRTUAL_ENV/lib -name "_rinterface_cffi_api*.so" | head -1)
patchelf --force-rpath \
    --set-rpath "$CONDA_PREFIX/lib/R/lib:$CONDA_PREFIX/lib" "$CFFI_SO"
```

Without patchelf you can use `LD_PRELOAD` as a per-process workaround:

```bash
LD_PRELOAD="$CONDA_PREFIX/lib/R/lib/libR.so" python my_script.py
```

### `.Rprofile` interference

If an ancestor directory contains an `.Rprofile` (e.g. a sibling project's
`renv/activate.R`), R will source it at startup, potentially modifying
`.libPaths()` in unexpected ways. Disable `.Rprofile` loading when running tests
or one-off scripts:

```bash
R_PROFILE_USER=/dev/null python my_script.py
```

To inject a custom R library path without touching `.Rprofile`:

```bash
R_LIBS_USER=/path/to/my/Rlib python my_script.py
# or equivalently via scibex:
python -c "import scibex as ib; ib.setup(lib_loc='/path/to/my/Rlib'); ..."
```

### Encoder models require keras / tensorflow

`method="encoder"` (the default) downloads model weights on first use via the
`basilisk`-managed Python environment inside the Ibex R package. If keras is not
available, use the fast geometric baseline instead:

```python
ib.tl.ibex(mdata, chain="Heavy", method="geometric")
```

## From source

```bash
git clone https://github.com/Qile0317/scibex
cd scibex
uv pip install -e ".[dev]"
```
