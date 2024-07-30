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
    spin: ffsim.Spin = ffsim.Spin.ALPHA_AND_BETA,
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

    expected_phase = phase_11 if spin & ffsim.Spin.ALPHA else 1
    np.testing.assert_allclose(
        np.vdot(state_11, gate(state_11, norb, (2, 0))),
        expected_phase,
        rtol=rtol,
        atol=atol,
    )

    actual_mat = np.zeros((2, 2), dtype=complex)
    expected_mat = mat if spin & ffsim.Spin.ALPHA else np.eye(2)
    for (a, state_a), (b, state_b) in itertools.product(
        enumerate([state_01, state_10]), repeat=2
    ):
        actual_mat[a, b] = np.vdot(state_a, gate(state_b, norb, (1, 0)))
    np.testing.assert_allclose(actual_mat, expected_mat, rtol=rtol, atol=atol)

    # test beta
    state_10 = ffsim.slater_determinant(norb=norb, occupied_orbitals=([], [i]))
    state_01 = ffsim.slater_determinant(norb=norb, occupied_orbitals=([], [j]))
    state_11 = ffsim.slater_determinant(norb=norb, occupied_orbitals=([], [i, j]))

    expected_phase = phase_11 if spin & ffsim.Spin.BETA else 1
    np.testing.assert_allclose(
        np.vdot(state_11, gate(state_11, norb, (0, 2))),
        expected_phase,
        rtol=rtol,
        atol=atol,
    )

    actual_mat = np.zeros((2, 2), dtype=complex)
    expected_mat = mat if spin & ffsim.Spin.BETA else np.eye(2)
    for (a, state_a), (b, state_b) in itertools.product(
        enumerate([state_01, state_10]), repeat=2
    ):
        actual_mat[a, b] = np.vdot(state_a, gate(state_b, norb, (0, 1)))
    np.testing.assert_allclose(actual_mat, expected_mat, rtol=rtol, atol=atol)


def assert_has_two_orbital_matrix_spinless(
    gate: Callable[[np.ndarray, int, int], np.ndarray],
    target_orbs: tuple[int, int],
    mat: np.ndarray,
    phase_00: complex,
    phase_11: complex,
    norb: int,
    rtol: float = 1e-7,
    atol: float = 0.0,
):
    state_00 = ffsim.slater_determinant(norb=norb, occupied_orbitals=[])
    np.testing.assert_allclose(
        np.vdot(state_00, gate(state_00, norb, 0)), phase_00, rtol=rtol, atol=atol
    )

    i, j = target_orbs

    state_10 = ffsim.slater_determinant(norb=norb, occupied_orbitals=[i])
    state_01 = ffsim.slater_determinant(norb=norb, occupied_orbitals=[j])
    state_11 = ffsim.slater_determinant(norb=norb, occupied_orbitals=[i, j])

    np.testing.assert_allclose(
        np.vdot(state_11, gate(state_11, norb, 2)), phase_11, rtol=rtol, atol=atol
    )

    actual_mat = np.zeros((2, 2), dtype=complex)
    for (a, state_a), (b, state_b) in itertools.product(
        enumerate([state_01, state_10]), repeat=2
    ):
        actual_mat[a, b] = np.vdot(state_a, gate(state_b, norb, 1))
    np.testing.assert_allclose(actual_mat, mat, rtol=rtol, atol=atol)


@pytest.mark.parametrize("norb, spin", ffsim.testing.generate_norb_spin(range(4)))
def test_apply_givens_rotation_matrix_spinful(norb: int, spin: ffsim.Spin):
    """Test Givens rotation matrix."""
    rng = np.random.default_rng()

    def mat(theta: float, phi: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.exp(1j * phi) * np.sin(theta)
        return np.array([[c, -s.conjugate()], [s, c]])

    phase_00 = 1
    phase_11 = 1

    for _ in range(3):
        theta = rng.uniform(-10, 10)
        phi = rng.uniform(-10, 10)
        for i, j in itertools.combinations(range(norb), 2):
            for target_orbs in [(i, j), (j, i)]:
                assert_has_two_orbital_matrix(
                    lambda vec, norb, nelec: ffsim.apply_givens_rotation(
                        vec,
                        theta,
                        phi=phi,
                        target_orbs=target_orbs,
                        norb=norb,
                        nelec=nelec,
                        spin=spin,
                    ),
                    target_orbs=target_orbs,
                    mat=mat(theta, phi),
                    phase_00=phase_00,
                    phase_11=phase_11,
                    norb=norb,
                    spin=spin,
                )


@pytest.mark.parametrize("norb", range(4))
def test_apply_givens_rotation_matrix_spinless(norb: int):
    """Test Givens rotation matrix, spinless."""
    rng = np.random.default_rng()

    def mat(theta: float, phi: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.exp(1j * phi) * np.sin(theta)
        return np.array([[c, -s.conjugate()], [s, c]])

    phase_00 = 1
    phase_11 = 1

    for _ in range(3):
        theta = rng.uniform(-10, 10)
        phi = rng.uniform(-10, 10)
        for i, j in itertools.combinations(range(norb), 2):
            for target_orbs in [(i, j), (j, i)]:
                assert_has_two_orbital_matrix_spinless(
                    lambda vec, norb, nelec: ffsim.apply_givens_rotation(
                        vec,
                        theta,
                        phi=phi,
                        target_orbs=target_orbs,
                        norb=norb,
                        nelec=nelec,
                    ),
                    target_orbs=target_orbs,
                    mat=mat(theta, phi),
                    phase_00=phase_00,
                    phase_11=phase_11,
                    norb=norb,
                )


def test_apply_givens_rotation_definition():
    """Test definition of complex Givens in terms of real Givens and phases."""
    norb = 5
    nelec = (3, 2)
    rng = np.random.default_rng()
    theta = rng.uniform(-10, 10)
    phi = rng.uniform(-10, 10)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=rng)

    # apply complex givens rotation
    result = ffsim.apply_givens_rotation(
        vec, theta, phi=phi, target_orbs=(1, 2), norb=norb, nelec=nelec
    )

    # get expected result using real givens rotation and phases
    expected = ffsim.apply_num_interaction(
        vec, -phi, target_orb=1, norb=norb, nelec=nelec
    )
    expected = ffsim.apply_givens_rotation(
        expected, theta, target_orbs=(1, 2), norb=norb, nelec=nelec
    )
    expected = ffsim.apply_num_interaction(
        expected, phi, target_orb=1, norb=norb, nelec=nelec
    )

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, spin", ffsim.testing.generate_norb_spin(range(4)))
def test_apply_tunneling_interaction_matrix_spinful(norb: int, spin: ffsim.Spin):
    """Test tunneling interaction matrix."""
    rng = np.random.default_rng()

    def mat(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, 1j * s], [1j * s, c]])

    phase_00 = 1
    phase_11 = 1

    for _ in range(3):
        theta = rng.uniform(-10, 10)
        for i, j in itertools.combinations(range(norb), 2):
            for target_orbs in [(i, j), (j, i)]:
                assert_has_two_orbital_matrix(
                    lambda vec, norb, nelec: ffsim.apply_tunneling_interaction(
                        vec,
                        theta,
                        target_orbs=target_orbs,
                        norb=norb,
                        nelec=nelec,
                        spin=spin,
                    ),
                    target_orbs=target_orbs,
                    mat=mat(theta),
                    phase_00=phase_00,
                    phase_11=phase_11,
                    norb=norb,
                    spin=spin,
                )


@pytest.mark.parametrize("norb", range(4))
def test_apply_tunneling_interaction_matrix_spinless(norb: int):
    """Test tunneling interaction matrix."""
    rng = np.random.default_rng()

    def mat(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, 1j * s], [1j * s, c]])

    phase_00 = 1
    phase_11 = 1

    for _ in range(3):
        theta = rng.uniform(-10, 10)
        for i, j in itertools.combinations(range(norb), 2):
            for target_orbs in [(i, j), (j, i)]:
                assert_has_two_orbital_matrix_spinless(
                    lambda vec, norb, nelec: ffsim.apply_tunneling_interaction(
                        vec,
                        theta,
                        target_orbs=target_orbs,
                        norb=norb,
                        nelec=nelec,
                    ),
                    target_orbs=target_orbs,
                    mat=mat(theta),
                    phase_00=phase_00,
                    phase_11=phase_11,
                    norb=norb,
                )


@pytest.mark.parametrize(
    "norb, nelec, spin", ffsim.testing.generate_norb_nelec_spin(range(4))
)
def test_apply_num_interaction_spinful(
    norb: int, nelec: tuple[int, int], spin: ffsim.Spin
):
    """Test applying number interaction."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    vec = np.array(ffsim.random.random_state_vector(dim, seed=rng))
    theta = rng.uniform(-10, 10)
    for target_orb in range(norb):
        result = ffsim.apply_num_interaction(
            vec, theta, target_orb, norb=norb, nelec=nelec, spin=spin
        )
        generator = theta * ffsim.number_operator(target_orb, spin=spin)
        linop = ffsim.linear_operator(generator, norb=norb, nelec=nelec)
        expected = scipy.sparse.linalg.expm_multiply(1j * linop, vec, traceA=1j * theta)
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nocc(range(4)))
def test_apply_num_interaction_spinless(norb: int, nelec: int):
    """Test applying number interaction, spinless."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    vec = np.array(ffsim.random.random_state_vector(dim, seed=rng))
    theta = rng.uniform(-10, 10)
    for target_orb in range(norb):
        result = ffsim.apply_num_interaction(
            vec, theta, target_orb, norb=norb, nelec=nelec
        )
        generator = theta * ffsim.number_operator(target_orb, spin=ffsim.Spin.ALPHA)
        linop = ffsim.linear_operator(generator, norb=norb, nelec=(nelec, 0))
        expected = scipy.sparse.linalg.expm_multiply(1j * linop, vec, traceA=1j * theta)
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "norb, nelec, spin", ffsim.testing.generate_norb_nelec_spin(range(4))
)
def test_apply_num_num_interaction_spinful(
    norb: int, nelec: tuple[int, int], spin: ffsim.Spin
):
    """Test applying number-number interaction."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    vec = np.array(ffsim.random.random_state_vector(dim, seed=rng))
    theta = rng.uniform(-10, 10)
    for i, j in itertools.combinations(range(norb), 2):
        for m, n in [(i, j), (j, i)]:
            result = ffsim.apply_num_num_interaction(
                vec, theta, (m, n), norb=norb, nelec=nelec, spin=spin
            )
            coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
            if spin & ffsim.Spin.ALPHA:
                coeffs[
                    (ffsim.cre_a(m), ffsim.des_a(m), ffsim.cre_a(n), ffsim.des_a(n))
                ] = theta
            if spin & ffsim.Spin.BETA:
                coeffs[
                    (ffsim.cre_b(m), ffsim.des_b(m), ffsim.cre_b(n), ffsim.des_b(n))
                ] = theta
            generator = ffsim.FermionOperator(coeffs)
            linop = ffsim.linear_operator(generator, norb=norb, nelec=nelec)
            expected = scipy.sparse.linalg.expm_multiply(
                1j * linop, vec, traceA=1j * theta
            )
            np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nocc(range(4)))
def test_apply_num_num_interaction_spinless(norb: int, nelec: int):
    """Test applying number-number interaction, spinless."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    vec = np.array(ffsim.random.random_state_vector(dim, seed=rng))
    theta = rng.uniform(-10, 10)
    for i, j in itertools.combinations(range(norb), 2):
        for m, n in [(i, j), (j, i)]:
            result = ffsim.apply_num_num_interaction(
                vec, theta, (m, n), norb=norb, nelec=nelec
            )
            coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
            coeffs[(ffsim.cre_a(m), ffsim.des_a(m), ffsim.cre_a(n), ffsim.des_a(n))] = (
                theta
            )
            generator = ffsim.FermionOperator(coeffs)
            linop = ffsim.linear_operator(generator, norb=norb, nelec=(nelec, 0))
            expected = scipy.sparse.linalg.expm_multiply(
                1j * linop, vec, traceA=1j * theta
            )
            np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(4)))
def test_apply_num_num_interaction_eigenvalues(norb: int, nelec: tuple[int, int]):
    """Test eigenvalues of number-number interaction."""
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


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(4)))
def test_apply_on_site_interaction(norb: int, nelec: tuple[int, int]):
    """Test applying on-site number-number interaction."""
    dim = ffsim.dim(norb, nelec)
    rng = np.random.default_rng()
    vec = np.array(ffsim.random.random_state_vector(dim, seed=rng))
    theta = rng.uniform(-10, 10)
    for i in range(norb):
        result = ffsim.apply_on_site_interaction(vec, theta, i, norb=norb, nelec=nelec)
        generator = ffsim.FermionOperator(
            {
                (
                    ffsim.cre_a(i),
                    ffsim.des_a(i),
                    ffsim.cre_b(i),
                    ffsim.des_b(i),
                ): theta
            }
        )
        linop = ffsim.linear_operator(generator, norb=norb, nelec=nelec)
        expected = scipy.sparse.linalg.expm_multiply(1j * linop, vec, traceA=1j * theta)
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(4)))
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


@pytest.mark.parametrize("norb, spin", ffsim.testing.generate_norb_spin(range(4)))
def test_apply_hop_gate_matrix_spinful(norb: int, spin: ffsim.Spin):
    """Test applying hop gate matrix."""
    rng = np.random.default_rng()

    def mat(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [s, c]])

    phase_00 = 1
    phase_11 = -1

    for _ in range(3):
        theta = rng.uniform(-10, 10)
        for i, j in itertools.combinations(range(norb), 2):
            for target_orbs in [(i, j), (j, i)]:
                assert_has_two_orbital_matrix(
                    lambda vec, norb, nelec: ffsim.apply_hop_gate(
                        vec,
                        theta,
                        target_orbs=target_orbs,
                        norb=norb,
                        nelec=nelec,
                        spin=spin,
                    ),
                    target_orbs=target_orbs,
                    mat=mat(theta),
                    phase_00=phase_00,
                    phase_11=phase_11,
                    norb=norb,
                    spin=spin,
                )


@pytest.mark.parametrize("norb", range(4))
def test_apply_hop_gate_matrix_spinless(norb: int):
    """Test applying hop gate matrix, spinless."""
    rng = np.random.default_rng()

    def mat(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [s, c]])

    phase_00 = 1
    phase_11 = -1

    for _ in range(3):
        theta = rng.uniform(-10, 10)
        for i, j in itertools.combinations(range(norb), 2):
            for target_orbs in [(i, j), (j, i)]:
                assert_has_two_orbital_matrix_spinless(
                    lambda vec, norb, nelec: ffsim.apply_hop_gate(
                        vec,
                        theta,
                        target_orbs=target_orbs,
                        norb=norb,
                        nelec=nelec,
                    ),
                    target_orbs=target_orbs,
                    mat=mat(theta),
                    phase_00=phase_00,
                    phase_11=phase_11,
                    norb=norb,
                )


@pytest.mark.parametrize("norb, spin", ffsim.testing.generate_norb_spin(range(4)))
def test_apply_fsim_gate_matrix_spinful(norb: int, spin: ffsim.Spin):
    """Test applying fSim gate matrix."""
    rng = np.random.default_rng()

    def mat(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -1j * s], [-1j * s, c]])

    phase_00 = 1

    for _ in range(3):
        theta = rng.uniform(-10, 10)
        phi = rng.uniform(-10, 10)
        phase_11 = np.exp(-1j * phi)
        for i, j in itertools.combinations(range(norb), 2):
            for target_orbs in [(i, j), (j, i)]:
                assert_has_two_orbital_matrix(
                    lambda vec, norb, nelec: ffsim.apply_fsim_gate(
                        vec,
                        theta,
                        phi,
                        target_orbs=target_orbs,
                        norb=norb,
                        nelec=nelec,
                        spin=spin,
                    ),
                    target_orbs=target_orbs,
                    mat=mat(theta),
                    phase_00=phase_00,
                    phase_11=phase_11,
                    norb=norb,
                    spin=spin,
                )


@pytest.mark.parametrize("norb", range(4))
def test_apply_fsim_gate_matrix_spinless(norb: int):
    """Test applying fSim gate matrix, spinless."""
    rng = np.random.default_rng()

    def mat(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -1j * s], [-1j * s, c]])

    phase_00 = 1

    for _ in range(3):
        theta = rng.uniform(-10, 10)
        phi = rng.uniform(-10, 10)
        phase_11 = np.exp(-1j * phi)
        for i, j in itertools.combinations(range(norb), 2):
            for target_orbs in [(i, j), (j, i)]:
                assert_has_two_orbital_matrix_spinless(
                    lambda vec, norb, nelec: ffsim.apply_fsim_gate(
                        vec,
                        theta,
                        phi,
                        target_orbs=target_orbs,
                        norb=norb,
                        nelec=nelec,
                    ),
                    target_orbs=target_orbs,
                    mat=mat(theta),
                    phase_00=phase_00,
                    phase_11=phase_11,
                    norb=norb,
                )


@pytest.mark.parametrize("norb, spin", ffsim.testing.generate_norb_spin(range(4)))
def test_apply_fswap_gate_matrix_spinful(norb: int, spin: ffsim.Spin):
    """Test applying fSWAP gate matrix."""

    mat01 = np.array([[0, 1], [1, 0]])

    phase_00 = 1
    phase_11 = -1

    for i, j in itertools.combinations(range(norb), 2):
        for target_orbs in [(i, j), (j, i)]:
            assert_has_two_orbital_matrix(
                lambda vec, norb, nelec: ffsim.apply_fswap_gate(
                    vec,
                    target_orbs=target_orbs,
                    norb=norb,
                    nelec=nelec,
                    spin=spin,
                ),
                target_orbs=target_orbs,
                mat=mat01,
                phase_00=phase_00,
                phase_11=phase_11,
                norb=norb,
                spin=spin,
            )


@pytest.mark.parametrize("norb", range(4))
def test_apply_fswap_gate_matrix_spinless(norb: int):
    """Test applying fSWAP gate matrix, spinless."""

    mat01 = np.array([[0, 1], [1, 0]])

    phase_00 = 1
    phase_11 = -1

    for i, j in itertools.combinations(range(norb), 2):
        for target_orbs in [(i, j), (j, i)]:
            assert_has_two_orbital_matrix_spinless(
                lambda vec, norb, nelec: ffsim.apply_fswap_gate(
                    vec,
                    target_orbs=target_orbs,
                    norb=norb,
                    nelec=nelec,
                ),
                target_orbs=target_orbs,
                mat=mat01,
                phase_00=phase_00,
                phase_11=phase_11,
                norb=norb,
            )
