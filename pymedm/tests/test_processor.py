import jax
import pandas
import pytest

from pymedm import processor_access, processor_availability
from pymedm.pmedm import GPU_ERROR, TPU_ERROR

####################################################################################
#           These test should be both platform & GPU-presence agnostic.
####################################################################################


def test_cpu_access():
    known = True
    observed = processor_access(processor="cpu")
    assert observed == known


def test_gpu_access():
    try:
        jax.devices("gpu")[0]
        known = True
    except RuntimeError as e:
        if str(e).startswith(GPU_ERROR):
            known = False
        else:
            raise e
    observed = processor_access(processor="gpu")
    assert observed == known


def test_tpu_access():
    try:
        jax.devices("tpu")[0]
        known = True
    except RuntimeError as e:
        if str(e).startswith(TPU_ERROR):
            known = False
        else:
            raise e
    observed = processor_access(processor="tpu")
    assert observed == known


def test_bad_processor():
    with pytest.raises(RuntimeError, match="Unknown backend mpu"):
        processor_access(processor="mpu")


def test_processor_availability_return():
    known_columns = ["device", "available", "count"]
    known_index = [0, 1, 2]

    observed = processor_availability()

    assert isinstance(observed, pandas.DataFrame)
    assert observed.columns.to_list() == known_columns
    assert observed.index.to_list() == known_index


def test_processor_availability_print():
    assert isinstance(processor_availability(indent_print=1), pandas.DataFrame)
