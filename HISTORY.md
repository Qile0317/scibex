# History

## 0.1.0b3 (2026-06-22)

* `backend="python"` parameter for `scibex.ibex_matrix` and `scibex.tl.ibex`, loads Keras encoder models directly in Python
* Keras and TensorFlow are now core dependencies.
* `scibex.setup(lib_loc=...)` now prepends to R's `.libPaths()` so all R package lookups (Ibex, immApex, etc.) respect the custom path for all inputs.

## 0.1.0b2 (2026-06-15)

* Minor documentation improvements and an additional tutorial notebook.

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
