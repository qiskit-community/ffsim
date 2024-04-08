# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for diagonal Coulomb evolution gate."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

import ffsim


@pytest.mark.parametrize(
    "norb, nelec, z_representation",
    [
        (norb, nelec, z_representation)
        for (norb, nelec), z_representation in itertools.product(
            ffsim.testing.generate_norb_nelec(range(5)), [False, True]
        )
    ],
)
def test_random_diag_coulomb_mat(
    norb: int, nelec: tuple[int, int], z_representation: bool
):
    """Test random orbital rotation circuit gives correct output state."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        mat_alpha_beta = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        time = rng.uniform(-10, 10)

        # Test without separate alpha-beta matrix
        gate = ffsim.qiskit.DiagCoulombEvolutionJW(
            mat, time, z_representation=z_representation
        )
        small_vec = ffsim.random.random_statevector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )
        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )
        expected = ffsim.apply_diag_coulomb_evolution(
            small_vec,
            mat,
            time,
            norb=norb,
            nelec=nelec,
            z_representation=z_representation,
        )
        np.testing.assert_allclose(result, expected)

        # Test with separate alpha-beta matrix
        gate = ffsim.qiskit.DiagCoulombEvolutionJW(
            mat, time, mat_alpha_beta=mat_alpha_beta, z_representation=z_representation
        )
        small_vec = ffsim.random.random_statevector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )
        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )
        expected = ffsim.apply_diag_coulomb_evolution(
            small_vec,
            mat,
            time,
            norb=norb,
            nelec=nelec,
            mat_alpha_beta=mat_alpha_beta,
            z_representation=z_representation,
        )
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "norb, nelec, z_representation",
    [
        (norb, nelec, z_representation)
        for (norb, nelec), z_representation in itertools.product(
            ffsim.testing.generate_norb_nelec(range(5)), [False, True]
        )
    ],
)
def test_inverse(norb: int, nelec: tuple[int, int], z_representation: bool):
    """Test inverse."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        mat_alpha_beta = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        time = rng.uniform(-10, 10)

        # Test without separate alpha-beta matrix
        gate = ffsim.qiskit.DiagCoulombEvolutionJW(
            mat, time, z_representation=z_representation
        )
        vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            ffsim.random.random_statevector(dim, seed=rng), norb=norb, nelec=nelec
        )
        statevec = Statevector(vec).evolve(gate)
        statevec = statevec.evolve(gate.inverse())
        np.testing.assert_allclose(np.array(statevec), vec)

        # Test with separate alpha-beta matrix
        gate = ffsim.qiskit.DiagCoulombEvolutionJW(
            mat, time, mat_alpha_beta=mat_alpha_beta, z_representation=z_representation
        )
        vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            ffsim.random.random_statevector(dim, seed=rng), norb=norb, nelec=nelec
        )
        statevec = Statevector(vec).evolve(gate)
        statevec = statevec.evolve(gate.inverse())
        np.testing.assert_allclose(np.array(statevec), vec)
