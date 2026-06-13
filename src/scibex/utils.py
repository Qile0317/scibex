from __future__ import annotations

import os

_IBEX_MIN_VERSION = (1, 3, 2)


def _installed_ibex_version() -> tuple[int, ...] | None:
    """Return the installed Ibex version as an int tuple, or None if not installed."""
    from rpy2.robjects.packages import importr, isinstalled

    if not isinstalled("Ibex"):
        return None
    utils_r, base_r = importr("utils"), importr("base")
    ver_str = str(list(base_r.as_character(utils_r.packageVersion("Ibex")))[0])
    return tuple(int(x) for x in ver_str.split("."))


def install_r_deps(
    lib_loc: str | None = None,
    use_renv: bool = False,
    force: bool = False,
    upgrade: bool = False,
    verbose: bool = True,
) -> None:
    """Install the Ibex R package and its dependencies.

    Fetches Ibex from GitHub (``BorchLab/Ibex@devel``) via the R ``remotes``
    package.  Also installs ``callr``, which basilisk requires to run the
    encoder in a subprocess isolated from any pre-existing Python session
    (necessary when calling scibex from a Jupyter notebook via rpy2).

    Skips packages that are already installed unless ``force=True``, so
    repeated calls are fast.

    Parameters
    ----------
    lib_loc:
        R library directory to install into.  ``None`` uses R's default
        ``.libPaths()[1]``, i.e. the first writable library on the path.
        Ignored when an renv project is active (renv manages the library).
    use_renv:
        If ``True``, initialise an ``renv`` project in the current working
        directory before installing.  This creates an isolated R library
        scoped to the project (named ``"scibex"`` in the renv metadata),
        which keeps Ibex and its dependencies separate from the system
        library.  Requires the ``renv`` R package to be available.
    force:
        If ``True``, reinstall all packages even if already present.
    upgrade:
        If ``False`` (default), never upgrade existing Ibex dependencies
        (passes ``upgrade="never"`` to ``remotes::install_github``).
        If ``True``, always upgrade (passes ``upgrade="always"``).  The
        default avoids the interactive prompt that hangs notebooks.
    verbose:
        If ``True`` (default), print a status line for each package being
        installed and a confirmation message when everything is already
        up-to-date.  Set to ``False`` to silence all output.

    Examples
    --------
    Quickstart — install into the default R library:

    >>> import scibex as ib
    >>> ib.install_r_deps()

    Install into a project-local renv:

    >>> ib.install_r_deps(use_renv=True)

    Install into a specific directory:

    >>> ib.install_r_deps(lib_loc="/path/to/my/R/library")
    """
    from rpy2.robjects.packages import importr, isinstalled
    from rpy2.robjects.vectors import StrVector

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    if use_renv:
        renv_r = importr("renv")
        renv_r.init(project=os.getcwd(), **{"project.name": "scibex"})

    # Detect whether an renv project is active in the current R session.
    # When renv is active, install.packages() is intercepted but may not
    # persist across renv::restore(); renv::install() + renv::snapshot()
    # integrates correctly and persists packages in the lockfile.
    try:
        _renv_r_check = importr("renv")
        _renv_active: bool = bool(list(_renv_r_check.is_active())[0])
    except Exception:
        _renv_active = False

    installed: list[str] = []

    if _renv_active:
        renv_r = importr("renv")
        for pkg in ("remotes", "callr"):
            if force or not isinstalled(pkg):
                _log(f"Installing {pkg} (via renv)...")
                renv_r.install(pkg, prompt=False)
                installed.append(pkg)
        _ibex_ver = _installed_ibex_version()
        _ibex_outdated = _ibex_ver is None or _ibex_ver < _IBEX_MIN_VERSION
        if force or _ibex_outdated:
            min_str = ".".join(map(str, _IBEX_MIN_VERSION))
            if _ibex_ver is not None and _ibex_ver < _IBEX_MIN_VERSION:
                _log(f"Upgrading Ibex {'.'.join(map(str, _ibex_ver))} → >={min_str} (via renv)...")
            else:
                _log("Installing Ibex from BorchLab/Ibex@devel (via renv)...")
            renv_r.install("BorchLab/Ibex@devel", prompt=False)
            installed.append("Ibex")
        if installed:
            # Persist newly installed packages to renv.lock so they survive
            # future renv::restore() calls without re-running install_r_deps().
            renv_r.snapshot(packages=StrVector(installed), prompt=False)
    else:
        utils_r = importr("utils")
        cran_kwargs: dict = {"repos": "https://cloud.r-project.org"}
        if lib_loc is not None:
            cran_kwargs["lib"] = lib_loc

        for pkg in ("remotes", "callr"):
            if force or not isinstalled(pkg):
                _log(f"Installing {pkg}...")
                utils_r.install_packages(pkg, **cran_kwargs)
                installed.append(pkg)

        remotes = importr("remotes")
        gh_kwargs: dict = {"upgrade": "always" if upgrade else "never"}
        if lib_loc is not None:
            gh_kwargs["lib"] = lib_loc

        _ibex_ver = _installed_ibex_version()
        _ibex_outdated = _ibex_ver is None or _ibex_ver < _IBEX_MIN_VERSION
        if force or _ibex_outdated:
            min_str = ".".join(map(str, _IBEX_MIN_VERSION))
            if _ibex_ver is not None and _ibex_ver < _IBEX_MIN_VERSION:
                _log(f"Upgrading Ibex {'.'.join(map(str, _ibex_ver))} → >={min_str}...")
            else:
                _log("Installing Ibex from BorchLab/Ibex@devel...")
            remotes.install_github("BorchLab/Ibex@devel", **gh_kwargs)
            installed.append("Ibex")

    if not installed:
        _log("All R dependencies already installed.")
