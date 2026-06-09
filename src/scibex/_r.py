from functools import cache

from rpy2.robjects.packages import importr


@cache
def ibex_r():
    """Cached Ibex R package instance (loaded once per process)."""
    return importr("Ibex")
