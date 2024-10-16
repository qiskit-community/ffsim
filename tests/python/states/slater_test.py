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

import numpy as np
import pytest

import ffsim
from ffsim.states.bitstring import BitstringType


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nocc([4, 5]))
def test_slater_determinant_amplitudes_spinless(norb: int, nelec: int):
    """Test computing Slater determinant amplitudes, spinless."""
    rng = np.random.default_rng(3725)

    dim = ffsim.dim(norb, nelec)
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)

    for occupied_orbitals in itertools.combinations(range(norb), nelec):
        bitstrings = ffsim.addresses_to_strings(range(dim), norb=norb, nelec=nelec)
        actual = ffsim.slater_determinant_amplitudes(
            bitstrings, norb, occupied_orbitals, orbital_rotation
        )
        expected = ffsim.slater_determinant(
            norb, occupied_orbitals=occupied_orbitals, orbital_rotation=orbital_rotation
        )
        ffsim.testing.assert_allclose_up_to_global_phase(actual, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec([3, 4]))
def test_slater_determinant_amplitudes_spinful(norb: int, nelec: int):
    """Test computing Slater determinant amplitudes, spinful."""
    rng = np.random.default_rng(3725)

    dim_a, dim_b = ffsim.dims(norb, nelec)
    n_alpha, n_beta = nelec
    orb_rot_a = ffsim.random.random_unitary(norb, seed=rng)
    orb_rot_b = ffsim.random.random_unitary(norb, seed=rng)

    bitstrings_a = ffsim.addresses_to_strings(range(dim_a), norb=norb, nelec=n_alpha)
    bitstrings_b = ffsim.addresses_to_strings(range(dim_b), norb=norb, nelec=n_beta)
    bitstrings = list(zip(*itertools.product(bitstrings_a, bitstrings_b)))

    for occ_a in itertools.combinations(range(norb), n_alpha):
        for occ_b in itertools.combinations(range(norb), n_beta):
            actual = ffsim.slater_determinant_amplitudes(
                bitstrings, norb, (occ_a, occ_b), (orb_rot_a, orb_rot_b)
            )
            expected = ffsim.slater_determinant(
                norb,
                occupied_orbitals=(occ_a, occ_b),
                orbital_rotation=(orb_rot_a, orb_rot_b),
            )
            ffsim.testing.assert_allclose_up_to_global_phase(actual, expected)
            actual = ffsim.slater_determinant_amplitudes(
                bitstrings, norb, (occ_a, occ_b), (None, orb_rot_b)
            )
            expected = ffsim.slater_determinant(
                norb,
                occupied_orbitals=(occ_a, occ_b),
                orbital_rotation=(None, orb_rot_b),
            )
            ffsim.testing.assert_allclose_up_to_global_phase(actual, expected)


@pytest.mark.parametrize(
    "norb, nelec, bitstring_type",
    [
        (norb, nelec, bitstring_type)
        for (norb, nelec), bitstring_type in itertools.product(
            ffsim.testing.generate_norb_nelec(range(1, 5)), BitstringType
        )
    ],
)
def test_sample_slater_determinant_spinful(
    norb: int, nelec: tuple[int, int], bitstring_type: BitstringType
):
    """Test sample Slater determinant, spinful."""
    rng = np.random.default_rng(1234)
    shots = 1000
    for _ in range(min(2, ffsim.dim(norb, nelec))):
        rotation_a = ffsim.random.random_unitary(norb, seed=rng)
        rotation_b = ffsim.random.random_unitary(norb, seed=rng)
        occupied_orbitals = ffsim.testing.random_occupied_orbitals(
            norb, nelec, seed=rng
        )
        rdm_a, rdm_b = ffsim.slater_determinant_rdms(
            norb, occupied_orbitals, (rotation_a, rotation_b)
        )
        vec = ffsim.slater_determinant(
            norb, occupied_orbitals, (rotation_a, rotation_b)
        )
        test_distribution = np.abs(vec) ** 2
        samples = ffsim.sample_slater_determinant(
            (rdm_a, rdm_b),
            norb,
            nelec,
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
def test_sample_slater_determinant_spinless(
    norb: int, nelec: int, bitstring_type: BitstringType
):
    """Test sample Slater determinant, spinless."""
    rng = np.random.default_rng(1234)
    shots = 1000
    rotation = ffsim.random.random_unitary(norb, seed=rng)
    for occupied_orbitals in itertools.combinations(range(norb), nelec):
        rdm = ffsim.slater_determinant_rdms(norb, occupied_orbitals, rotation, rank=1)
        vec = ffsim.slater_determinant(norb, occupied_orbitals, rotation)
        test_distribution = np.abs(vec) ** 2
        samples = ffsim.sample_slater_determinant(
            rdm, norb, nelec, shots=shots, bitstring_type=bitstring_type, seed=rng
        )
        addresses = ffsim.strings_to_addresses(samples, norb, nelec)
        indices, counts = np.unique(addresses, return_counts=True)
        assert np.sum(counts) == shots
        empirical_distribution = np.zeros(ffsim.dim(norb, nelec), dtype=float)
        empirical_distribution[indices] = counts / shots
        assert np.sum(np.sqrt(test_distribution * empirical_distribution)) > 0.99


def test_sample_slater_determinant_large():
    """Test sample Slater determinant for a larger number of orbitals."""
    norb = 6
    nelec = (3, 2)

    rng = np.random.default_rng(1234)
    shots = 5000
    rotation_a = ffsim.random.random_unitary(norb, seed=rng)
    rotation_b = ffsim.random.random_unitary(norb, seed=rng)
    occupied_orbitals = [(0, 2, 3), (2, 4)]
    rdm_a, rdm_b = ffsim.slater_determinant_rdms(
        norb, occupied_orbitals, (rotation_a, rotation_b)
    )
    vec = ffsim.slater_determinant(norb, occupied_orbitals, (rotation_a, rotation_b))
    test_distribution = np.abs(vec) ** 2
    samples = ffsim.sample_slater_determinant(
        (rdm_a, rdm_b), norb, nelec, shots=shots, seed=rng
    )
    addresses = ffsim.strings_to_addresses(samples, norb, nelec)
    indices, counts = np.unique(addresses, return_counts=True)
    assert np.sum(counts) == shots
    empirical_distribution = np.zeros(ffsim.dim(norb, nelec), dtype=float)
    empirical_distribution[indices] = counts / shots
    assert np.sum(np.sqrt(test_distribution * empirical_distribution)) > 0.99


def test_sample_slater_determinant_restrict():
    """Test sample Slater determinant with subset of orbitals."""
    norb = 8
    nelec = (4, 3)

    shots = 10
    occupied_orbitals = [(0, 2, 3, 5), (2, 3, 4)]
    rdm_a, rdm_b = ffsim.slater_determinant_rdms(norb, occupied_orbitals)
    samples = ffsim.sample_slater_determinant(
        (rdm_a, rdm_b), norb, nelec, orbs=([1, 2, 5], [3, 4, 5]), shots=shots
    )
    assert samples == ["011110"] * 10
