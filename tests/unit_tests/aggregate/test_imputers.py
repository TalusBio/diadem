"""Test our RT alignment module."""
import logging

import numpy as np

from diadem.aggregate.imputers import MatrixFactorizationModel, MFImputer


def test_model():
    """Test the base pytorch model."""
    model = MatrixFactorizationModel(n_peptides=10, n_runs=5, n_factors=3, rng=1)
    assert model.peptide_factors.shape == (10, 3)
    assert model.run_factors.shape == (3, 5)
    assert model().shape == (10, 5)


def test_imputer(caplog):
    """Test the imputer."""
    caplog.set_level(logging.INFO)
    rng = np.random.default_rng(42)
    model = MFImputer(n_factors=3, task="retention time", rng=1)
    peptide_factors = rng.random((100, 3))
    run_factors = rng.random((3, 20))
    mat = peptide_factors @ run_factors
    assert mat.shape == (100, 20)

    # Without NaNs:
    pred = model.fit(mat).transform()
    np.testing.assert_allclose(pred, mat, rtol=1e-5)

    # With NaNs:
    mask = rng.binomial(1, 0.1, size=mat.shape).astype(bool)
    missing = mat.copy()
    missing[mask] = np.nan
    pred = model.fit_transform(missing)
    np.testing.assert_allclose(pred, mat, rtol=1e-5)


def test_search_factors(caplog):
    """Test searching for the best number of factors."""
    caplog.set_level(logging.INFO)
    rng = np.random.default_rng(42)
    model = MFImputer(n_factors=None, task="retention time", rng=1)
    peptide_factors = rng.random((101, 3))
    run_factors = rng.random((3, 20))
    mat = peptide_factors @ run_factors
    assert mat.shape == (101, 20)
    assert model.n_factors is None

    model = model.search_factors(mat, (2, 3, 4))
    assert model.n_factors == 3
