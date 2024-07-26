# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Slater determinant functions."""

from __future__ import annotations

import itertools
import random

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
def test_slater_sampler(
    norb: int, nelec: tuple[int, int], bitstring_type: BitstringType
):
    """Test Slater determinant sampler."""

    rng = np.random.default_rng(1234)
    shots = 1000
    n_a, n_b = nelec

    rotation_a = ffsim.random.random_unitary(norb, seed=rng)
    rotation_b = ffsim.random.random_unitary(norb, seed=rng)
    for alpha_orbitals in random.sample(
        list(itertools.combinations(range(norb), n_a)),
        max(1, ffsim.dim(norb, n_a) // 2),
    ):
        for beta_orbitals in random.sample(
            list(itertools.combinations(range(norb), n_b)),
            max(1, ffsim.dim(norb, n_b) // 2),
        ):
            occupied_orbitals = (alpha_orbitals, beta_orbitals)
            rdm_a, rdm_b = ffsim.slater_determinant_rdms(
                norb, occupied_orbitals, (rotation_a, rotation_b)
            )

            test_distribution = (
                np.abs(
                    ffsim.slater_determinant(
                        norb, occupied_orbitals, (rotation_a, rotation_b)
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
    shots = 1000
    rotation = ffsim.random.random_unitary(norb, seed=rng)

    for occupied_orbitals in random.sample(
        list(itertools.combinations(range(norb), nelec)),
        max(1, ffsim.dim(norb, nelec) // 2),
    ):
        rdm = ffsim.slater_determinant_rdms(norb, occupied_orbitals, rotation, rank=1)
        test_distribution = (
            np.absolute(ffsim.slater_determinant(norb, occupied_orbitals, rotation))
            ** 2
        )
        samples = ffsim.sample_slater(
            rdm, norb, nelec, shots=shots, bitstring_type=bitstring_type, seed=rng
        )
        addresses = ffsim.strings_to_addresses(samples, norb, nelec)
        indices, counts = np.unique(addresses, return_counts=True)
        empirical_distribution = np.zeros(ffsim.dim(norb, nelec), dtype=float)
        empirical_distribution[indices] = counts / np.sum(counts)

        assert np.sum(np.sqrt(test_distribution * empirical_distribution)) > 0.99
