# scibex

[![PyPI version](https://img.shields.io/pypi/v/scibex.svg)](https://pypi.org/project/scibex/)
[![Documentation Status](https://readthedocs.org/projects/scibex/badge/?version=latest)](https://scibex.readthedocs.io/en/latest/?version=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**scibex** brings [Ibex](https://github.com/BorchLab/Ibex) BCR embeddings into the [scverse](https://scverse.org) ecosystem. It wraps the R Ibex package via rpy2 and stores results directly in `AnnData.obsm`, making it a drop-in complement to [scirpy](https://scirpy.scverse.org) for B-cell receptor analysis.

Ibex encodes CDR3 (or CDR1+2+3) amino acid sequences from paired heavy and light chains using convolutional/variational autoencoders or a fast geometric transform. The resulting low-dimensional embeddings can be combined with gene expression data for multimodal single-cell analysis.

---

## Features

- **scirpy-native**: reads chain sequences from `obsm["chain_indices"]`; writes embeddings back to `obsm`
- **Heavy and light chains**: embed each independently then combine downstream
- **Multiple models**: geometric baseline, CNN autoencoder, VAE, and expanded CDR1+2+3 variants
- **Multiple encodings**: Atchley factors, Kidera factors, Cruciani properties, MSWHIM, tScales, one-hot
- **No manual sequence handling**: `scibex.tl.ibex(mdata, ...)` does the full extract → embed → store pipeline

---

## Installation

```bash
pip install scibex
```

scibex wraps the [Ibex R package](https://github.com/BorchLab/Ibex) via rpy2.
Install the R dependency from Python:

```python
import scibex as ib
ib.install_r_deps()                            # into R's default library
ib.install_r_deps(lib_loc="/path/to/my/Rlib")  # into a specific directory
ib.install_r_deps(force=True)                  # force-reinstall everything
```

<!-- This also installs `callr`, which basilisk needs to run the encoder in an
isolated subprocess (required when calling scibex from a Jupyter notebook). -->

Or directly in R:

```r
remotes::install_github("BorchLab/Ibex@devel")
```

If Ibex is in a non-default R library, call `ib.setup(lib_loc=...)` **once**
before any embedding call:

```python
ib.setup(lib_loc="/path/to/my/Rlib")
```

See the [Installation docs](docs/installation.md) for R environment
troubleshooting (conda ABI mismatches, `.Rprofile` interference, keras setup).

---

## Quick start

```python
import scirpy as ir
import scibex as ib

# Load a scirpy MuData (chain_indices must already be populated)
mdata = ir.datasets.stephenson2021_5k()

# Embed heavy-chain CDR3 sequences → stored in mdata["airr"].obsm["X_ibex_heavy"]
ib.tl.ibex(mdata, chain="Heavy", key_added="X_ibex_heavy")

# Embed light-chain CDR3 sequences → stored in mdata["airr"].obsm["X_ibex_light"]
ib.tl.ibex(mdata, chain="Light", key_added="X_ibex_light")
```

Switch `encoder_input` or `encoder_model` for different representations:

```python
ib.tl.ibex(
    mdata,
    chain="Heavy",
    method="encoder",
    encoder_model="VAE",
    encoder_input="kideraFactors",
    key_added="X_ibex_heavy",
)
```

<!--
For a fast geometric baseline (no model download needed):

```python
ib.tl.ibex(mdata, chain="Heavy", method="geometric", key_added="X_ibex_heavy")
``` 
-->

If you only have a list of sequences (e.g. from a custom pipeline), use the low-level function directly:

```python
embedding = ib.ibex_matrix(
    ["CARDLVSYGMDVW", "CAKGGQIFHFSSGFYFDFW"],
    chain="Heavy",
    method="encoder",
)  # returns np.ndarray of shape [N, D]
```

---

## Tutorial

A complete end-to-end tutorial on the Stephenson 2021 COVID-19 dataset (5k BCR cells) is available in
the [Tutorials](tutorials/) section (`docs/notebooks/tutorial_5k_bcr.ipynb`).

It covers:

- Loading a scirpy `MuData`
- Embedding heavy and light chains with `scibex.tl.ibex`
- Visualising the embedding space as a UMAP
- Training a logistic-regression classifier to predict patient outcome from paired BCR embeddings

---

## API overview

| Function | Description |
| --- | --- |
| `scibex.tl.ibex(adata, ...)` | Embed BCR sequences in a scirpy `AnnData`/`MuData`; stores result in `obsm` |
| `scibex.ibex_matrix(seqs, ...)` | Low-level: embed a list of CDR3 strings, returns `[N, D]` numpy array |

**Key parameters for `tl.ibex`:**

| Parameter | Options | Default |
| --- | --- | --- |
| `chain` | `"Heavy"`, `"Light"` | `"Heavy"` |
| `method` | `"encoder"`, `"geometric"` | `"encoder"` |
| `encoder_model` | `"CNN"`, `"VAE"`, `"CNN.EXP"`, `"VAE.EXP"` | `"VAE"` |
| `encoder_input` | `"atchleyFactors"`, `"kideraFactors"`, `"crucianiProperties"`, `"MSWHIM"`, `"tScales"`, `"OHE"` | `"atchleyFactors"` |
| `species` | `"Human"`, `"Mouse"` | `"Human"` |
| `key_added` | any string | `"X_ibex"` |

---

## Acknowledgements

scibex is a Python interface to the [Ibex R package](https://github.com/BorchLab/Ibex). If you use scibex in your work, please cite the original Ibex publication.

- PyPI: <https://pypi.org/project/scibex/>
- Documentation: <https://scibex.readthedocs.io>
- License: MIT
