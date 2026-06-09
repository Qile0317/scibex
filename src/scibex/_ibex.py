import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import StrVector, pandas2ri

from ._r import ibex_r


def ibex_matrix(
    sequences: list[str],
    chain: str = "Heavy",
    method: str = "geometric",
    encoder_model: str = "VAE",
    encoder_input: str = "atchleyFactors",
    geometric_theta: float = 3.14159265,
    species: str = "Human",
    verbose: bool = False,
) -> np.ndarray:
    """Embed CDR3 sequences using the Ibex R package.

    Returns an [N, D] float array; rows correspond to input sequences in order.
    """
    r_result = ibex_r().Ibex_matrix(
        StrVector(sequences),
        chain=chain,
        method=method,
        encoder_model=encoder_model,
        encoder_input=encoder_input,
        geometric_theta=geometric_theta,
        species=species,
        verbose=verbose,
    )
    with (ro.default_converter + pandas2ri.converter).context():
        df = ro.conversion.get_conversion().rpy2py(r_result)
    return df.to_numpy()
