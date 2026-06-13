# History

## 0.1.0b1 (2026-06-12)

* First public pre-release on PyPI.
* Python interface to the Ibex BCR embedding R package for scverse single-cell analysis.
* `scibex.tl.ibex` — embed BCR sequences from scirpy `AnnData`/`MuData` into `obsm`.
* `scibex.ibex_matrix` — low-level embedding of a plain sequence list.
* Supports geometric, CNN, and VAE encoders; CDR3-only and expanded CDR1+2+3 (EXP) variants.
* `strategy` parameter (`"lenient"` / `"strict"`) for handling partial CDR missingness in EXP models.
* `fill_value` and `verbose` options for missing-sequence handling.
* Central type aliases (`_types.py`) for all shared `Literal` annotations.
* Zensical documentation site; ReadTheDocs integration.
