# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for sampling Slater determinants."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import ffsim
from ffsim.states.bitstring import BitstringType


@pytest.mark.parametrize(
    "norb, nelec, bitstring_type",
    [
        (norb, nelec, bitstring_type)
        for (norb, nelec), bitstring_type in itertools.product(
            ffsim.testing.generate_norb_nelec(range(1, 5)), BitstringType
        )
    ],
)
def test_sample_slater_spinful(
    norb: int, nelec: tuple[int, int], bitstring_type: BitstringType
):
    """Test sample Slater, spinful."""
    rng = np.random.default_rng(1234)
    shots = 1000
    for _ in range(min(2, ffsim.dim(norb, nelec))):
        rotation_a = ffsim.random.random_unitary(norb, seed=rng)
        rotation_b = ffsim.random.random_unitary(norb, seed=rng)
        occupied_orbitals = ffsim.testing.random_occupied_orbitals(
            norb, nelec, seed=rng
        )
        vec = ffsim.slater_determinant(
            norb, occupied_orbitals, (rotation_a, rotation_b)
        )
        test_distribution = np.abs(vec) ** 2
        samples = ffsim.sample_slater(
            norb,
            occupied_orbitals,
            (rotation_a, rotation_b),
            shots=shots,
            bitstring_type=bitstring_type,
            seed=rng,
        )
        addresses = ffsim.strings_to_addresses(samples, norb, nelec)
        indices, counts = np.unique(addresses, return_counts=True)
        assert np.sum(counts) == shots
        empirical_distribution = np.zeros(ffsim.dim(norb, nelec), dtype=float)
        empirical_distribution[indices] = counts / shots
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
def test_sample_slater_spinless(norb: int, nelec: int, bitstring_type: BitstringType):
    """Test sample Slater, spinless."""
    rng = np.random.default_rng(1234)
    shots = 1000

    rotation = ffsim.random.random_unitary(norb, seed=rng)
    for occupied_orbitals in itertools.combinations(range(norb), nelec):
        vec = ffsim.slater_determinant(norb, occupied_orbitals, rotation)
        test_distribution = np.abs(vec) ** 2
        samples = ffsim.sample_slater(
            norb,
            occupied_orbitals,
            rotation,
            shots=shots,
            bitstring_type=bitstring_type,
            seed=rng,
        )
        addresses = ffsim.strings_to_addresses(samples, norb, nelec)
        indices, counts = np.unique(addresses, return_counts=True)
        assert np.sum(counts) == shots
        empirical_distribution = np.zeros(ffsim.dim(norb, nelec), dtype=float)
        empirical_distribution[indices] = counts / shots
        assert np.sum(np.sqrt(test_distribution * empirical_distribution)) > 0.99


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (6, (3, 2)),
        (6, (5, 4)),  # high filling, triggers hole trick
    ],
)
@pytest.mark.parametrize("real", [True, False])
def test_sample_slater_large(norb: int, nelec: tuple[int, int], real: bool):
    """Test sample Slater for a larger number of orbitals."""
    rng = np.random.default_rng(1234)
    shots = 5000
    random_unitary = (
        ffsim.random.random_orthogonal if real else ffsim.random.random_unitary
    )
    rotation_a = random_unitary(norb, seed=rng)
    rotation_b = random_unitary(norb, seed=rng)
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)
    vec = ffsim.slater_determinant(norb, occupied_orbitals, (rotation_a, rotation_b))
    test_distribution = np.abs(vec) ** 2
    samples = ffsim.sample_slater(
        norb, occupied_orbitals, (rotation_a, rotation_b), shots=shots, seed=rng
    )
    addresses = ffsim.strings_to_addresses(samples, norb, nelec)
    indices, counts = np.unique(addresses, return_counts=True)
    assert np.sum(counts) == shots
    empirical_distribution = np.zeros(ffsim.dim(norb, nelec), dtype=float)
    empirical_distribution[indices] = counts / shots
    assert np.sum(np.sqrt(test_distribution * empirical_distribution)) > 0.99


@pytest.mark.parametrize(
    "norb, nelec, orbs",
    [
        (8, (4, 3), ([1, 2, 5], [3, 4, 5])),
        (8, (6, 5), ([0, 3, 6], [2, 4, 7])),  # high filling, triggers hole trick
    ],
)
def test_sample_slater_restrict(
    norb: int, nelec: tuple[int, int], orbs: tuple[list[int], list[int]]
):
    """Test sample Slater with subset of orbitals."""
    rng = np.random.default_rng(1234)
    shots = 10
    orbs_a, orbs_b = orbs
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)
    occ_a, occ_b = occupied_orbitals
    alpha_str = "".join("1" if o in occ_a else "0" for o in reversed(orbs_a))
    beta_str = "".join("1" if o in occ_b else "0" for o in reversed(orbs_b))
    expected = f"{beta_str}{alpha_str}"
    samples = ffsim.sample_slater(norb, occupied_orbitals, orbs=orbs, shots=shots)
    assert samples == [expected] * shots
