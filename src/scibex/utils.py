from __future__ import annotations

import os


def install_r_deps(
    lib_loc: str | None = None,
    use_renv: bool = False,
    force: bool = False,
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
    use_renv:
        If ``True``, initialise an ``renv`` project in the current working
        directory before installing.  This creates an isolated R library
        scoped to the project (named ``"scibex"`` in the renv metadata),
        which keeps Ibex and its dependencies separate from the system
        library.  Requires the ``renv`` R package to be available.
    force:
        If ``True``, reinstall all packages even if already present.

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

    if use_renv:
        renv = importr("renv")
        renv.init(project=os.getcwd(), **{"project.name": "scibex"})

    utils_r = importr("utils")
    cran_kwargs: dict = {"repos": "https://cloud.r-project.org"}
    if lib_loc is not None:
        cran_kwargs["lib"] = lib_loc

    for pkg in ("remotes", "callr"):
        if force or not isinstalled(pkg):
            utils_r.install_packages(pkg, **cran_kwargs)

    remotes = importr("remotes")
    gh_kwargs: dict = {}
    if lib_loc is not None:
        gh_kwargs["lib"] = lib_loc

    if force or not isinstalled("Ibex"):
        remotes.install_github("BorchLab/Ibex@devel", **gh_kwargs)
