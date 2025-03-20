# pymedm

**Penalized Maximum-Entropy Dasymetric Modeling (P-MEDM) in Python**

[![pipeline status](https://code.ornl.gov/likeness/pymedm/badges/develop/pipeline.svg?job=karma&key_text=pipeline:+develop&key_width=110)](https://code.ornl.gov/likeness/pymedm/-/commits/develop)
[![pipeline status](https://code.ornl.gov/likeness/pymedm/badges/main/pipeline.svg?job=karma&key_text=pipeline:+main&key_width=110)](https://code.ornl.gov/likeness/pymedm/-/commits/main)
[![coverage report](https://code.ornl.gov/likeness/pymedm/badges/develop/coverage.svg)](https://code.ornl.gov/likeness/pymedm/-/commits/develop)
[![Latest Release](https://code.ornl.gov/likeness/pymedm/-/badges/release.svg)](https://code.ornl.gov/likeness/pymedm/-/releases)

This is a _Work in Progress_ Python port of [PMEDMrcpp](https://bitbucket.org/jovtc/pmedmrcpp/src/master). 

## Important notes & announcements
#### Updated testing (01/2023)
A problem arose in 01/2023 with reproducing testing results. It is detailed in a [Wiki](https://code.ornl.gov/likeness/pymedm/-/wikis/Updating-testing-explanation-%5B01/Updating-testing-explanation). If further issues arise, please tag [#18](https://code.ornl.gov/likeness/pymedm/-/issues/18) and also add to the Wiki entry.
#### Windows users
Windows users who are interested in `pymedm` are encouraged to use the [`pmedm_legacy`](https://code.ornl.gov/likeness/pmedm_legacy) pacakge along with [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) due to compatibility and efficiency issues. 

## Installation, Development, & Contributing

* **See instructions [here](https://code.ornl.gov/groups/urbanpop-py/-/wikis/Installation,-Development,-&-Contributing).**

### NVIDIA CUDA Setup
Running `pymedm` on NVIDIA CUDA systems requires some additional libraries. Once the environment is installed: 

```
conda activate py312_pymedm
```

then

```
bash install_cuda_reqs.sh
```

## References

1. **Leyk, S., Nagle, N. N., & Buttenfield, B. P.** (2013). Maximum entropy dasymetric modeling for demographic small area estimation. Geographical Analysis, 45(3), 285-306.
2. **Nagle, N. N., Buttenfield, B. P., Leyk, S., & Spielman, S.** (2014). Dasymetric modeling and uncertainty. Annals of the Association of American Geographers, 104(1), 80-95.
