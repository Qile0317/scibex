from __future__ import annotations

import os


def install_r_deps(
    lib_loc: str | None = None,
    use_renv: bool = False,
) -> None:
    """Install the Ibex R package and its dependencies.

    Fetches Ibex from GitHub (``BorchLab/Ibex``) via the R ``remotes``
    package.  If ``remotes`` is not installed, it is installed automatically
    from CRAN before proceeding.

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
    if not isinstalled("remotes"):
        utils_r.install_packages("remotes", repos="https://cloud.r-project.org")

    remotes = importr("remotes")
    kwargs: dict = {}
    if lib_loc is not None:
        kwargs["lib"] = lib_loc
    remotes.install_github("BorchLab/Ibex", **kwargs)
