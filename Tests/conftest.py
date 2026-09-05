"""Reproducible numerical tests without GUI windows or leaked global state."""

import random

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

from BayesianDLL import Model, ObservedParameter, RandomParameter
from BayesianDLL.Distributions import Normal


@pytest.fixture(autouse=True)
def numerical_environment():
    dtype = torch.get_default_dtype()
    threads = torch.get_num_threads()
    rng = torch.random.get_rng_state()
    numpy_rng = np.random.get_state()
    python_rng = random.getstate()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    np.random.seed(2026)
    random.seed(2026)
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)
    torch.random.set_rng_state(rng)
    np.random.set_state(numpy_rng)
    random.setstate(python_rng)


@pytest.fixture
def normal_model():
    """Normal(0, 4) prior and unit-variance data: posterior N(1.6, 4/13)."""
    with Model() as model:
        mean = RandomParameter("mean", Normal(0.0, 4.0))
        ObservedParameter("data", Normal(mean, 1.0), torch.tensor([1.0, 2.0, 2.2]))
    return model
