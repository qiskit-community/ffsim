"""Test Slater sampler."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import ffsim
from ffsim.states.bitstring import BitstringType


def _empirical_distribution(bts_matrix, norb, nelec):
    indices = np.zeros(bts_matrix.shape[0], dtype=int)
    for i, bts in enumerate(bts_matrix):
        string = np.array2string(bts, separator="")[1:-1]
        index = ffsim.strings_to_indices([string], norb, nelec)[0]
        indices[i] = index

    unique_indices, counts = np.unique(indices, return_counts=True)

    probabilities = np.zeros(ffsim.dim(norb, nelec))

    probabilities[unique_indices] = counts
    probabilities /= np.sum(probabilities)

    return probabilities


@pytest.mark.parametrize(
    "norb, nelec, bitstring_type",
    [
        (norb, nelec, bitstring_type)
        for (norb, nelec), bitstring_type in itertools.product(
            ffsim.testing.generate_norb_nelec(range(1, 5)), BitstringType
        )
    ],
)
def test_slater_sampler(
    norb: int, nelec: tuple[int, int], bitstring_type: BitstringType
):
    """Test Slater determinant sampler."""

    rng = np.random.default_rng(1234)
    shots = 3000
    n_a, n_b = nelec

    rotation_a = ffsim.random.random_unitary(norb, seed=rng)
    rotation_b = ffsim.random.random_unitary(norb, seed=rng)
    rdm_a, rdm_b = ffsim.slater_determinant_rdms(
        norb, (range(n_a), range(n_b)), (rotation_a, rotation_b)
    )
    test_distribution = (
        np.abs(
            ffsim.slater_determinant(
                norb, (range(n_a), range(n_b)), (rotation_a, rotation_b)
            )
        )
        ** 2
    )
    samples = ffsim.sample_slater(
        (rdm_a, rdm_b),
        norb,
        nelec,
        shots=shots,
        bitstring_type=bitstring_type,
        seed=rng,
    )

    addresses = ffsim.strings_to_addresses(samples, norb, nelec)

    indices, counts = np.unique(addresses, return_counts=True)
    empirical_distribution = np.zeros(ffsim.dim(norb, nelec), dtype=float)
    empirical_distribution[indices] = counts / np.sum(counts)

    # empirical_distribution = _empirical_distribution(samples, norb, nelec)
    assert np.sum(np.sqrt(test_distribution * empirical_distribution)) > 0.99


@pytest.mark.parametrize(
    "norb, nelec, bitstring_type",
    [
        (norb, nelec, bitstring_type)
        for (norb, nelec), bitstring_type in itertools.product(
            ffsim.testing.generate_norb_nocc(range(1, 5)), BitstringType
        )
    ],
)
def test_slater_sampler_spinless(norb: int, nelec: int, bitstring_type: BitstringType):
    """Test Slater determinant sampler (spinless case)."""

    rng = np.random.default_rng(1234)
    shots = 3000
    rotation = ffsim.random.random_unitary(norb, seed=rng)
    rdm = ffsim.slater_determinant_rdms(norb, range(nelec), rotation, rank=1)
    test_distribution = (
        np.absolute(ffsim.slater_determinant(norb, range(nelec), rotation)) ** 2
    )
    samples = ffsim.sample_slater(
        rdm, norb, nelec, shots=shots, bitstring_type=bitstring_type, seed=rng
    )
    addresses = ffsim.strings_to_addresses(samples, norb, nelec)

    indices, counts = np.unique(addresses, return_counts=True)
    empirical_distribution = np.zeros(ffsim.dim(norb, nelec), dtype=float)
    empirical_distribution[indices] = counts / np.sum(counts)

    # empirical_distribution = _empirical_distribution(samples, norb, nelec)
    assert np.sum(np.sqrt(test_distribution * empirical_distribution)) > 0.99
