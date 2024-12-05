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
from typing import cast

import numpy as np
import pytest

import ffsim


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_hartree_fock_state_spinful(norb: int, nelec: tuple[int, int]):
    """Test Hartree-Fock state."""
    vec = ffsim.hartree_fock_state(norb, nelec)
    dim = ffsim.dim(norb, nelec)
    assert vec.shape == (dim,)
    assert vec[0] == 1
    assert all(vec[1:] == 0)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nocc(range(5)))
def test_hartree_fock_state_spinless(norb: int, nelec: int):
    """Test Hartree-Fock state."""
    vec = ffsim.hartree_fock_state(norb, nelec)
    dim = ffsim.dim(norb, nelec)
    assert vec.shape == (dim,)
    assert vec[0] == 1
    assert all(vec[1:] == 0)


@pytest.mark.parametrize("norb, nocc", ffsim.testing.generate_norb_nocc(range(5)))
def test_slater_determinant_spinless(norb: int, nocc: int):
    """Test Slater determinant with same rotation for both spins."""
    rng = np.random.default_rng()

    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nocc, seed=rng)

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    eigs, orbital_rotation = np.linalg.eigh(one_body_tensor)
    eig = sum(eigs[occupied_orbitals])
    state = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )

    hamiltonian = ffsim.contract.one_body_linop(
        one_body_tensor, norb=norb, nelec=(nocc, 0)
    )
    np.testing.assert_allclose(hamiltonian @ state, eig * state)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_slater_determinant_same_rotation(norb: int, nelec: tuple[int, int]):
    """Test Slater determinant with same rotation for both spins."""
    rng = np.random.default_rng()

    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)
    occ_a, occ_b = occupied_orbitals

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    eigs, orbital_rotation = np.linalg.eigh(one_body_tensor)
    eig = sum(eigs[occ_a]) + sum(eigs[occ_b])
    state = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )

    hamiltonian = ffsim.contract.one_body_linop(one_body_tensor, norb=norb, nelec=nelec)
    np.testing.assert_allclose(hamiltonian @ state, eig * state)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_slater_determinant_diff_rotation(norb: int, nelec: tuple[int, int]):
    """Test Slater determinant with different rotations for each spin."""
    rng = np.random.default_rng()

    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)
    occ_a, occ_b = occupied_orbitals

    orbital_rotation_a = ffsim.random.random_unitary(norb, seed=rng)
    orbital_rotation_b = ffsim.random.random_unitary(norb, seed=rng)

    state = ffsim.slater_determinant(
        norb,
        occupied_orbitals,
        orbital_rotation=(orbital_rotation_a, orbital_rotation_b),
    )
    state_a = ffsim.slater_determinant(
        norb,
        (occ_a, []),
        orbital_rotation=orbital_rotation_a,
    )
    state_b = ffsim.slater_determinant(
        norb,
        ([], occ_b),
        orbital_rotation=orbital_rotation_b,
    )

    np.testing.assert_allclose(state, np.kron(state_a, state_b))


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_slater_determinant_rdm1s_same_rotation(norb: int, nelec: tuple[int, int]):
    """Test Slater determinant 1-RDM."""
    rng = np.random.default_rng()

    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)

    vec = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )
    rdm = ffsim.slater_determinant_rdms(
        norb,
        occupied_orbitals,
        orbital_rotation=orbital_rotation,
    )
    expected = ffsim.rdms(vec, norb, nelec)

    np.testing.assert_allclose(rdm, expected, atol=1e-12)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_slater_determinant_rdm1s_diff_rotation(norb: int, nelec: tuple[int, int]):
    """Test Slater determinant 1-RDM."""
    rng = np.random.default_rng()

    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)
    orbital_rotation_a = ffsim.random.random_unitary(norb, seed=rng)
    orbital_rotation_b = ffsim.random.random_unitary(norb, seed=rng)

    vec = ffsim.slater_determinant(
        norb,
        occupied_orbitals,
        orbital_rotation=(orbital_rotation_a, orbital_rotation_b),
    )
    expected = ffsim.rdms(vec, norb, nelec)

    rdm = ffsim.slater_determinant_rdms(
        norb,
        occupied_orbitals,
        orbital_rotation=(orbital_rotation_a, orbital_rotation_b),
    )
    np.testing.assert_allclose(rdm, expected, atol=1e-12)

    rdm = ffsim.slater_determinant_rdms(
        norb,
        occupied_orbitals,
        orbital_rotation=np.stack([orbital_rotation_a, orbital_rotation_b]),
    )
    np.testing.assert_allclose(rdm, expected, atol=1e-12)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nocc(range(5)))
def test_slater_determinant_rdm1s_spinless(norb: int, nelec: int):
    """Test Slater determinant 1-RDM, spinless."""
    rng = np.random.default_rng()

    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)

    vec = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )
    rdm = ffsim.slater_determinant_rdms(
        norb,
        occupied_orbitals,
        orbital_rotation=orbital_rotation,
    )
    expected = ffsim.rdms(vec, norb, (nelec, 0), spin_summed=True)

    np.testing.assert_allclose(rdm, expected, atol=1e-12)


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
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec([3, 4]))
def test_slater_determinant_amplitudes_spinful(norb: int, nelec: tuple[int, int]):
    """Test computing Slater determinant amplitudes, spinful."""
    rng = np.random.default_rng(3725)

    dim_a, dim_b = ffsim.dims(norb, nelec)
    n_alpha, n_beta = nelec
    orb_rot_a = ffsim.random.random_unitary(norb, seed=rng)
    orb_rot_b = ffsim.random.random_unitary(norb, seed=rng)

    bitstrings_a = ffsim.addresses_to_strings(range(dim_a), norb=norb, nelec=n_alpha)
    bitstrings_b = ffsim.addresses_to_strings(range(dim_b), norb=norb, nelec=n_beta)
    bitstrings = cast(
        tuple[tuple[int], tuple[int]],
        tuple(zip(*itertools.product(bitstrings_a, bitstrings_b))),
    )

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
