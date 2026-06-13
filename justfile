# Justfile for scibex

# Show available commands
list:
    @just --list

# One-time setup: recompile rpy2 so it links against the conda env's libR.so.
# Only needed on Linux when using a conda-installed R alongside a system R.
# Not required for non-conda R installs — standard `uv sync` suffices there.
#
# If patchelf is available (conda install -c conda-forge patchelf), the CFFI
# extension is patched in-place so the correct libR.so is baked into the rpath
# and LD_PRELOAD is no longer needed at runtime.
setup-r:
    uv sync --extra test
    bash -c 'VIRTUAL_ENV="$(pwd)/.venv" LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib/R/lib" uv pip install --reinstall --no-binary rpy2 rpy2'
    bash -c 'CFFI_SO="$(find "$(pwd)/.venv/lib" -name "_rinterface_cffi_api*.so" 2>/dev/null | head -1)"; [ -z "$CFFI_SO" ] && exit 0; if command -v patchelf >/dev/null 2>&1; then echo "Patching rpath: $CFFI_SO"; patchelf --force-rpath --set-rpath "$CONDA_PREFIX/lib/R/lib:$CONDA_PREFIX/lib" "$CFFI_SO"; echo "Done — rpy2 will now load conda R without LD_PRELOAD"; else echo "patchelf not found; LD_PRELOAD fallback active in just recipes"; echo "For a permanent fix: conda install -c conda-forge patchelf && just setup-r"; fi'

# LD_PRELOAD fallback for the conda R + system R ABI conflict (Linux only).
# When a conda R and a system R coexist, rpy2's compiled CFFI extension may resolve
# libR.so to the system R via its rpath.  Preloading the conda libR.so forces the
# dynamic linker to use it instead.  Empty string on macOS or when not in a conda env
# (LD_PRELOAD="" is a no-op; macOS ignores LD_PRELOAD entirely).
# This is only a fallback — `just setup-r` with patchelf makes it unnecessary.
_os := `uname -s`
_libR := if _os == "Linux" { if env_var_or_default("CONDA_PREFIX", "") != "" { env_var_or_default("CONDA_PREFIX", "") + "/lib/R/lib/libR.so" } else { "" } } else { "" }

# Pytest wrapper: runs tests with LD_PRELOAD set for the conda R ABI fix.
_pytest *ARGS:
    LD_PRELOAD={{_libR}} uv run --python=3.13 --extra test pytest {{ARGS}}

# Run all the formatting, linting, and testing commands
qa:
    uv run --python=3.13 --extra test ruff format .
    uv run --python=3.13 --extra test ruff check . --fix
    uv run --python=3.13 --extra test ruff check --select I --fix .
    uv run --python=3.13 --extra test ty check .
    just _pytest .

# Run all the tests for all the supported Python versions
testall:
    LD_PRELOAD={{_libR}} uv run --python=3.10 --extra test pytest
    LD_PRELOAD={{_libR}} uv run --python=3.11 --extra test pytest
    LD_PRELOAD={{_libR}} uv run --python=3.12 --extra test pytest
    LD_PRELOAD={{_libR}} uv run --python=3.13 --extra test pytest

# Run all the tests, but allow for arguments to be passed
test *ARGS:
    @echo "Running with arg: {{ARGS}}"
    just _pytest {{ARGS}}

# Run all the tests, but on failure, drop into the debugger
pdb *ARGS:
    @echo "Running with arg: {{ARGS}}"
    LD_PRELOAD={{_libR}} uv run --python=3.13 --extra test pytest --pdb --maxfail=10 --pdbcls=IPython.terminal.debugger:TerminalPdb {{ARGS}}

# Run coverage, and build to HTML
coverage:
    LD_PRELOAD={{_libR}} uv run --python=3.13 --extra test coverage run -m pytest .
    uv run --python=3.13 --extra test coverage report -m
    uv run --python=3.13 --extra test coverage html

# Build the project, useful for checking that packaging is correct
build:
    rm -rf build
    rm -rf dist
    uv build

# Serve the docs locally with live reload at localhost:8000
docs-serve:
    uv run --extra docs zensical serve

# Build the docs to site/ (for local inspection)
docs-build:
    uv run --extra docs zensical build --clean

VERSION := `grep -m1 '^version' pyproject.toml | sed -E 's/version = "(.*)"/\1/'`

# Print the current version of the project
version:
    @echo "Current version is {{VERSION}}"

# Tag the current version in git and put to github
tag:
    echo "Tagging version v{{VERSION}}"
    git tag -a v{{VERSION}} -m "Creating version v{{VERSION}}"
    git push origin v{{VERSION}}

# remove all build, test, coverage and Python artifacts
clean: 
	clean-build
	clean-pyc
	clean-test

# remove build artifacts
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

# remove Python file artifacts
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

# remove test and coverage artifacts
clean-test:
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

# Publish to PyPI (manual alternative to GitHub Actions)
publish:
    uv build
    uv publish