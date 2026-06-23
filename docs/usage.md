# Usage

## Prerequisites

Before calling any embedding function, install the Ibex R dependency:

```python
import scibex as ib

ib.install_r_deps()
```

See [Installation](installation.md) for conda / custom library-path variants and
troubleshooting.

---

## Embedding BCR sequences in AnnData / MuData

`scibex.tl.ibex` is the main entry point.  It reads CDR sequences from a
scirpy-indexed `AnnData` or `MuData`, embeds them, and stores the result in
`obsm`.

```python
import scirpy as ir
import scibex as ib

mdata = ir.datasets.stephenson2021_5k()   # loads a scirpy MuData
ir.pp.index_chains(mdata)                 # required if not already done

# embed heavy-chain CDR3 → mdata["airr"].obsm["X_ibex_heavy"]
ib.tl.ibex(mdata, chain="Heavy", key_added="X_ibex_heavy")

# embed light-chain CDR3 → mdata["airr"].obsm["X_ibex_light"]
ib.tl.ibex(mdata, chain="Light", key_added="X_ibex_light")
```

### Choosing a model

| `method` | `encoder_model` | Description |
| --- | --- | --- |
| `"geometric"` | *(ignored)* | Fast transform; no download needed |
| `"encoder"` | `"VAE"` | Variational autoencoder on CDR3 |
| `"encoder"` | `"CNN"` | Convolutional autoencoder on CDR3 |
| `"encoder"` | `"VAE.EXP"` | VAE on CDR1+CDR2+CDR3 (expanded) |
| `"encoder"` | `"CNN.EXP"` | CNN on CDR1+CDR2+CDR3 (expanded) |

```python
ib.tl.ibex(
    mdata,
    chain="Heavy",
    method="encoder",
    encoder_model="VAE",
    encoder_input="kideraFactors",   # default: "atchleyFactors"
    species="Human",                  # or "Mouse"
    key_added="X_ibex_heavy",
)
```

### EXP models and the `strategy` parameter

EXP models (`VAE.EXP`, `CNN.EXP`) use CDR1, CDR2, and CDR3 together.  The
`strategy` parameter controls what happens when CDR1 or CDR2 is absent:

| `strategy` | CDR1/CDR2 missing | CDR3 missing |
| --- | --- | --- |
| `"lenient"` *(default)* | substitute `"NA"` → embed as `"NA-NA-CDR3"` | fill row with `fill_value` |
| `"strict"` | fill row with `fill_value` | fill row with `fill_value` |

```python
ib.tl.ibex(
    mdata,
    chain="Heavy",
    encoder_model="VAE.EXP",
    strategy="lenient",   # default
)
```

---

## Handling missing sequences

Cells that lack the requested chain or CDR data receive rows filled with
`fill_value` (default `0.0`).  Pass `fill_value=float("nan")` to restore
NaN-fill behaviour.  Use `verbose=True` to see a count of affected cells:

```python
ib.tl.ibex(
    mdata,
    chain="Heavy",
    method="geometric",
    fill_value=0.0,    # default; use float("nan") for NaN rows
    verbose=True,      # warns if any cells are missing chain data
    key_added="X_ibex_heavy",
)
```

---

## Python backend

By default, scibex embeds sequences by calling the Ibex R package, which
internally launches a basilisk-managed Python subprocess to run Keras.  The
`backend="python"` option short-circuits this chain and loads the `.keras`
encoder files directly in the current Python process — no R, no rpy2, no
subprocess.

The Python backend is the default — no extra install required. Pass `backend="python"` explicitly or omit it:

```python
import scibex as ib

seqs = ["CARDYW", "CARDSSGYW", "CARDTGYW"]
embedding = ib.ibex_matrix(seqs, chain="Heavy", encoder_model="CNN",
                            encoder_input="atchleyFactors", backend="python")
```

```python
ib.tl.ibex(mdata, chain="Heavy", encoder_model="VAE", backend="python",
           key_added="X_ibex_heavy")
```

Model files are downloaded on first use from Zenodo and cached in
`~/.cache/R/Ibex/` — the same directory the R package uses, so a prior
`ib.tl.ibex(..., backend="r")` call (or any previous run of the Ibex R
package) means the weights are already local.

> **Note:** `method="geometric"` is not supported with `backend="python"`.
> Pass `method="encoder"` (the default) or use `backend="r"`.

---

## Low-level: embedding a plain sequence list

Use `ibex_matrix` when you have sequences from outside a scirpy AnnData:

```python
import scibex as ib

seqs = ["CARDLVSYGMDVW", "CAKGGQIFHFSSGFYFDFW", None]  # None → fill row

embedding = ib.ibex_matrix(
    seqs,
    chain="Heavy",
    method="encoder",
    encoder_model="VAE",
    fill_value=0.0,    # default
    verbose=True,      # warns about the None entry
)
# embedding.shape → (3, D)
```

`None` entries in the sequence list receive rows filled with `fill_value`.
Raises `ValueError` if the list is empty or all entries are `None`.
