# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for basic gates."""

from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import pytest
import scipy.sparse.linalg

import ffsim


def assert_has_two_orbital_matrix(
    gate: Callable[[np.ndarray, int, tuple[int, int]], np.ndarray],
    target_orbs: tuple[int, int],
    mat: np.ndarray,
    phase_00: complex,
    phase_11: complex,
    norb: int,
    rtol: float = 1e-7,
    atol: float = 0.0,
):
    state_00 = ffsim.slater_determinant(norb=norb, occupied_orbitals=([], []))
    np.testing.assert_allclose(
        np.vdot(state_00, gate(state_00, norb, (0, 0))), phase_00, rtol=rtol, atol=atol
    )

    i, j = target_orbs

    # test alpha
    state_10 = ffsim.slater_determinant(norb=norb, occupied_orbitals=([i], []))
    state_01 = ffsim.slater_determinant(norb=norb, occupied_orbitals=([j], []))
    state_11 = ffsim.slater_determinant(norb=norb, occupied_orbitals=([i, j], []))

    np.testing.assert_allclose(
        np.vdot(state_11, gate(state_11, norb, (2, 0))), phase_11, rtol=rtol, atol=atol
    )

    actual_mat = np.zeros((2, 2), dtype=complex)
    for (a, state_a), (b, state_b) in itertools.product(
        enumerate([state_01, state_10]), repeat=2
    ):
        actual_mat[a, b] = np.vdot(state_a, gate(state_b, norb, (1, 0)))
    np.testing.assert_allclose(actual_mat, mat, rtol=rtol, atol=atol)

    # test beta
    state_10 = ffsim.slater_determinant(norb=norb, occupied_orbitals=([], [i]))
    state_01 = ffsim.slater_determinant(norb=norb, occupied_orbitals=([], [j]))
    state_11 = ffsim.slater_determinant(norb=norb, occupied_orbitals=([], [i, j]))

    np.testing.assert_allclose(
        np.vdot(state_11, gate(state_11, norb, (0, 2))), phase_11, rtol=rtol, atol=atol
    )

    actual_mat = np.zeros((2, 2), dtype=complex)
    for (a, state_a), (b, state_b) in itertools.product(
        enumerate([state_01, state_10]), repeat=2
    ):
        actual_mat[a, b] = np.vdot(state_a, gate(state_b, norb, (0, 1)))
    np.testing.assert_allclose(actual_mat, mat, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (2, (1, 1)),
        (4, (2, 2)),
        (5, (3, 2)),
    ],
)
def test_apply_givens_rotation(norb: int, nelec: tuple[int, int]):
    """Test Givens rotation."""
    dim = ffsim.dim(norb, nelec)
    rng = np.random.default_rng()
    vec = np.array(ffsim.random.random_statevector(dim, seed=rng))
    original_vec = vec.copy()
    theta = rng.uniform(-10, 10)
    for i, j in itertools.combinations(range(norb), 2):
        for target_orbs in [(i, j), (j, i)]:
            result = ffsim.apply_givens_rotation(
                vec, theta, target_orbs, norb=norb, nelec=nelec
            )
            generator = np.zeros((norb, norb))
            a, b = target_orbs
            generator[a, b] = theta
            generator[b, a] = -theta
            linop = ffsim.contract.one_body_linop(generator, norb=norb, nelec=nelec)
            expected = scipy.sparse.linalg.expm_multiply(
                linop, vec, traceA=np.sum(np.abs(generator))
            )
            np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(vec, original_vec)


def test_apply_givens_rotation_matrix():
    """Test Givens rotation matrix."""
    norb = 4
    rng = np.random.default_rng()

    def mat(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [s, c]])

    phase_00 = 1
    phase_11 = 1

    for _ in range(5):
        theta = rng.uniform(-10, 10)
        for i, j in itertools.combinations(range(norb), 2):
            for target_orbs in [(i, j), (j, i)]:
                assert_has_two_orbital_matrix(
                    lambda vec, norb, nelec: ffsim.apply_givens_rotation(
                        vec, theta, target_orbs=target_orbs, norb=norb, nelec=nelec
                    ),
                    target_orbs=target_orbs,
                    mat=mat(theta),
                    phase_00=phase_00,
                    phase_11=phase_11,
                    norb=norb,
                )


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (2, (1, 1)),
        (4, (2, 2)),
        (5, (3, 2)),
    ],
)
def test_apply_tunneling_interaction(norb: int, nelec: tuple[int, int]):
    """Test tunneling interaction."""
    dim = ffsim.dim(norb, nelec)
    rng = np.random.default_rng()
    vec = np.array(ffsim.random.random_statevector(dim, seed=rng))
    theta = rng.uniform(-10, 10)
    for i, j in itertools.combinations(range(norb), 2):
        for target_orbs in [(i, j), (j, i)]:
            result = ffsim.apply_tunneling_interaction(
                vec, theta, target_orbs, norb=norb, nelec=nelec
            )
            generator = np.zeros((norb, norb))
            generator[i, j] = theta
            generator[j, i] = theta
            linop = ffsim.contract.one_body_linop(generator, norb=norb, nelec=nelec)
            expected = scipy.sparse.linalg.expm_multiply(
                1j * linop, vec, traceA=np.sum(np.abs(generator))
            )
            np.testing.assert_allclose(result, expected)


def test_apply_tunneling_interaction_matrix():
    """Test tunneling interaction matrix."""
    norb = 4
    rng = np.random.default_rng()

    def mat(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, 1j * s], [1j * s, c]])

    phase_00 = 1
    phase_11 = 1

    for _ in range(5):
        theta = rng.uniform(-10, 10)
        for i, j in itertools.combinations(range(norb), 2):
            for target_orbs in [(i, j), (j, i)]:
                assert_has_two_orbital_matrix(
                    lambda vec, norb, nelec: ffsim.apply_tunneling_interaction(
                        vec, theta, target_orbs=target_orbs, norb=norb, nelec=nelec
                    ),
                    target_orbs=target_orbs,
                    mat=mat(theta),
                    phase_00=phase_00,
                    phase_11=phase_11,
                    norb=norb,
                )


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (2, (1, 1)),
        (4, (2, 2)),
        (5, (3, 2)),
    ],
)
def test_apply_num_interaction(norb: int, nelec: tuple[int, int]):
    """Test applying number interaction."""
    dim = ffsim.dim(norb, nelec)
    rng = np.random.default_rng()
    vec = np.array(ffsim.random.random_statevector(dim, seed=rng))
    theta = rng.uniform(-10, 10)
    for target_orb in range(norb):
        result = ffsim.apply_num_interaction(
            vec, theta, target_orb, norb=norb, nelec=nelec
        )
        generator = np.zeros((norb, norb))
        generator[target_orb, target_orb] = theta
        linop = ffsim.contract.one_body_linop(generator, norb=norb, nelec=nelec)
        expected = scipy.sparse.linalg.expm_multiply(
            1j * linop, vec, traceA=np.sum(np.abs(generator))
        )
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (2, (1, 1)),
        (3, (2, 1)),
    ],
)
def test_apply_num_num_interaction(norb: int, nelec: tuple[int, int]):
    """Test applying number interaction."""
    dim = ffsim.dim(norb, nelec)
    rng = np.random.default_rng()
    vec = np.array(ffsim.random.random_statevector(dim, seed=rng))
    theta = rng.uniform(-10, 10)
    for i, j in itertools.combinations(range(norb), 2):
        for target_orbs in [(i, j), (j, i)]:
            result = ffsim.apply_num_num_interaction(
                vec, theta, target_orbs, norb=norb, nelec=nelec
            )
            m, n = target_orbs
            generator = ffsim.FermionOperator(
                {
                    (
                        ffsim.cre_a(m),
                        ffsim.des_a(m),
                        ffsim.cre_a(n),
                        ffsim.des_a(n),
                    ): theta,
                    (
                        ffsim.cre_b(m),
                        ffsim.des_b(m),
                        ffsim.cre_b(n),
                        ffsim.des_b(n),
                    ): theta,
                }
            )
            linop = ffsim.linear_operator(generator, norb=norb, nelec=nelec)
            expected = scipy.sparse.linalg.expm_multiply(1j * linop, vec, traceA=theta)
            np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (2, (1, 1)),
        (4, (2, 2)),
        (5, (3, 2)),
    ],
)
def test_apply_num_num_interaction_eigenvalues(norb: int, nelec: tuple[int, int]):
    """Test applying number interaction."""
    rng = np.random.default_rng()
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec)
    vec = ffsim.slater_determinant(norb, occupied_orbitals)

    theta = rng.uniform(-10, 10)
    for i, j in itertools.combinations(range(norb), 2):
        for target_orbs in [(i, j), (j, i)]:
            result = ffsim.apply_num_num_interaction(
                vec,
                theta,
                target_orbs=target_orbs,
                norb=norb,
                nelec=nelec,
            )
            eig = 0.0
            for occ in occupied_orbitals:
                if i in occ and j in occ:
                    eig += theta
            expected = np.exp(1j * eig) * vec
            np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (2, (1, 1)),
        (4, (2, 2)),
        (5, (3, 2)),
    ],
)
def test_apply_num_op_prod(norb: int, nelec: tuple[int, int]):
    """Test applying number operator product interaction."""
    rng = np.random.default_rng()
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec)
    vec = ffsim.slater_determinant(norb, occupied_orbitals)

    theta = rng.uniform(-10, 10)
    for i, j in itertools.combinations_with_replacement(range(norb), 2):
        for spin_i, spin_j in itertools.product(range(2), repeat=2):
            target_orbs: tuple[set[int], set[int]] = (set(), set())
            target_orbs[spin_i].add(i)
            target_orbs[spin_j].add(j)
            alpha_orbs, beta_orbs = target_orbs
            result = ffsim.apply_num_op_prod_interaction(
                vec,
                theta,
                target_orbs=(list(alpha_orbs), list(beta_orbs)),
                norb=norb,
                nelec=nelec,
            )
            if i in occupied_orbitals[spin_i] and j in occupied_orbitals[spin_j]:
                eig = theta
            else:
                eig = 0
            expected = np.exp(1j * eig) * vec
            np.testing.assert_allclose(result, expected)


def test_apply_hop_gate_matrix():
    """Test applying hop gate matrix."""
    norb = 4
    rng = np.random.default_rng()

    def mat(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [s, c]])

    phase_00 = 1
    phase_11 = -1

    for _ in range(5):
        theta = rng.uniform(-10, 10)
        for i, j in itertools.combinations(range(norb), 2):
            for target_orbs in [(i, j), (j, i)]:
                assert_has_two_orbital_matrix(
                    lambda vec, norb, nelec: ffsim.apply_hop_gate(
                        vec, theta, target_orbs=target_orbs, norb=norb, nelec=nelec
                    ),
                    target_orbs=target_orbs,
                    mat=mat(theta),
                    phase_00=phase_00,
                    phase_11=phase_11,
                    norb=norb,
                )


def test_apply_fsim_gate_matrix():
    """Test applying fSim gate matrix."""
    norb = 4
    rng = np.random.default_rng()

    def mat(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -1j * s], [-1j * s, c]])

    phase_00 = 1

    for _ in range(5):
        theta = rng.uniform(-10, 10)
        phi = rng.uniform(-10, 10)
        phase_11 = np.exp(-1j * phi)
        for i, j in itertools.combinations(range(norb), 2):
            for target_orbs in [(i, j), (j, i)]:
                assert_has_two_orbital_matrix(
                    lambda vec, norb, nelec: ffsim.apply_fsim_gate(
                        vec, theta, phi, target_orbs=target_orbs, norb=norb, nelec=nelec
                    ),
                    target_orbs=target_orbs,
                    mat=mat(theta),
                    phase_00=phase_00,
                    phase_11=phase_11,
                    norb=norb,
                )
