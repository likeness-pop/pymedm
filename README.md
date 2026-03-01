# PyMEDM: Penalized Maximum-Entropy Dasymetric Modeling (P-MEDM) in Python

![tag](https://img.shields.io/github/v/release/likeness-pop/pymedm?include_prereleases&sort=semver)
[![PyPI version](https://badge.fury.io/py/pymedm.svg)](https://badge.fury.io/py/pymedm)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pymedm.svg)](https://anaconda.org/conda-forge/pymedm)

[![Continuous Integration](https://github.com/likeness-pop/pymedm/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/likeness-pop/pymedm/actions/workflows/continuous_integration.yml)
[![codecov](https://codecov.io/gh/likeness-pop/pymedm/branch/develop/graph/badge.svg)](https://codecov.io/gh/likeness-pop/pymedm)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This is a GPU-ready Python port of [PMEDMrcpp](https://bitbucket.org/jovtc/pmedmrcpp/src/master) via `jax` and `jaxopt`. Support for usage on Windows is not guaranteed.

## Installation

### Conda-forge (recommended)

The `pymedm` feedstock is available via the [conda-forge channel](https://github.com/conda-forge/pymedm-feedstock).

```
$ conda install --channel conda-forge pymedm
```

### PyPI

`pymedm` is available on the [Python Package Index](https://pypi.org/project/pymedm/).

```
$ pip install pymedm
```

### Source

#### Directly via GitHub + `pip`

```
$ pip install git+https://github.com/likeness-pop/pymedm.git@develop
```

#### Download + `pip`

Download the source distribution (``.tar.gz``) and decompress where desired. From that location:

```
$ pip install .
```

## Usage

* See usage examples in [`./notebooks/`](https://github.com/likeness-pop/pymedm/tree/main/notebooks)

## Development

1. Clone the repository to the desired location.
2. Install in editable mode
   * Navigate to where the repo was cloned locally.
   * Within that directory:
      ```
      $ pip install -e .
      ```
3. Open an Issue for discussion
4. In a branch off `develop`, implement update/bug fix/etc.
5. Create a minimal Pull Request with clear description linked back to the associated issue from (3.)
6. Wait for review from maintainers
7. Adjust as directed
8. Once merged, fetch down `origin/develop` and merge into the local `develop`
9. Delete the branch created in (4.)
10. Start over at (2.)

## References

1. **Leyk, S., Nagle, N. N., & Buttenfield, B. P.** (2013). Maximum entropy dasymetric modeling for demographic small area estimation. Geographical Analysis, 45(3), 285-306.
2. **Nagle, N. N., Buttenfield, B. P., Leyk, S., & Spielman, S.** (2014). Dasymetric modeling and uncertainty. Annals of the Association of American Geographers, 104(1), 80-95.

## Citation

If you find this package useful or use it an academic publication, please cite as:

* **Tuccillo, J.V. and Gaboardi, J.D.** (2025) *pymedm*. Computer Software. URL: https://github.com/likeness-pop/pymedm. DOI: [10.11578/dc.20250320.3](https://doi.org/10.11578/dc.20250320.3)
