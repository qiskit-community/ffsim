"""Test Slater sampler."""

from __future__ import annotations

import math

import numpy as np
import pytest

import ffsim


def _empirical_distribution(bts_matrix, norb, nelec):
    indices = np.zeros(bts_matrix.shape[0], dtype=int)
    for i, bts in enumerate(bts_matrix):
        string = np.array2string(bts, separator="")[1:-1]
        index = ffsim.strings_to_indices([string], norb, nelec)[0]
        indices[i] = index

    unique_indices, counts = np.unique(indices, return_counts=True)
    if isinstance(nelec, tuple):
        probabilities = np.zeros(math.comb(norb, nelec[0]) * math.comb(norb, nelec[1]))
    else:
        probabilities = np.zeros(math.comb(norb, nelec))

    probabilities[unique_indices] = counts
    probabilities /= np.sum(probabilities)

    return probabilities


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(1, 5)))
def test_slater_sampler(norb: int, nelec: tuple[int, int]):
    """Test Slater determinant sampler."""

    rng = np.random.default_rng(1234)
    n_samples = 3000
    n_a, n_b = nelec

    rotation_a = ffsim.random.random_unitary(norb, seed=rng)
    rotation_b = ffsim.random.random_unitary(norb, seed=rng)
    rdm_a = ffsim.slater_determinant_rdms(norb, range(n_a), rotation_a, rank=1)
    rdm_b = ffsim.slater_determinant_rdms(norb, range(n_b), rotation_b, rank=1)
    test_distribution = (
        np.abs(
            ffsim.slater_determinant(
                norb, (range(n_a), range(n_b)), (rotation_a, rotation_b)
            )
        )
        ** 2
    )
    samples = ffsim.sample_slater((rdm_a, rdm_b), n_samples, seed=rng)
    empirical_distribution = _empirical_distribution(samples, norb, nelec)
    assert np.sum(np.sqrt(test_distribution * empirical_distribution)) > 0.99


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nocc(range(1, 5)))
def test_slater_sampler_spinless(norb: int, nelec: int):
    """Test Slater determinant sampler (spinless case)."""

    rng = np.random.default_rng(1234)
    n_samples = 3000
    rotation = ffsim.random.random_unitary(norb, seed=rng)
    rdm = ffsim.slater_determinant_rdms(norb, range(nelec), rotation, rank=1)
    test_distribution = (
        np.absolute(ffsim.slater_determinant(norb, (range(nelec), []), rotation)) ** 2
    )
    samples = ffsim.sample_slater(rdm, n_samples, seed=rng)
    empirical_distribution = _empirical_distribution(samples, norb, nelec)
    assert np.sum(np.sqrt(test_distribution * empirical_distribution)) > 0.99
