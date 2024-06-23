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
def test_random_diag_coulomb_mat_spinful(
    norb: int, nelec: tuple[int, int], z_representation: bool
):
    """Test random diag Coulomb gate gives correct output state."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        mat_aa = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        mat_ab = rng.standard_normal((norb, norb))
        mat_bb = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        time = rng.uniform(-10, 10)

        # mat
        gate = ffsim.qiskit.DiagCoulombEvolutionJW(
            norb, mat_aa, time, z_representation=z_representation
        )
        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )
        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )
        expected = ffsim.apply_diag_coulomb_evolution(
            small_vec,
            mat_aa,
            time,
            norb=norb,
            nelec=nelec,
            z_representation=z_representation,
        )
        np.testing.assert_allclose(result, expected)

        # (mat_aa, mat_ab, mat_bb)
        gate = ffsim.qiskit.DiagCoulombEvolutionJW(
            norb,
            (mat_aa, mat_ab, mat_bb),
            time,
            z_representation=z_representation,
        )
        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )
        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )
        expected = ffsim.apply_diag_coulomb_evolution(
            small_vec,
            (mat_aa, mat_ab, mat_bb),
            time,
            norb=norb,
            nelec=nelec,
            z_representation=z_representation,
        )
        np.testing.assert_allclose(result, expected)

        # (mat_aa, None, mat_bb)
        gate = ffsim.qiskit.DiagCoulombEvolutionJW(
            norb,
            (mat_aa, None, mat_bb),
            time,
            z_representation=z_representation,
        )
        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )
        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )
        expected = ffsim.apply_diag_coulomb_evolution(
            small_vec,
            (mat_aa, None, mat_bb),
            time,
            norb=norb,
            nelec=nelec,
            z_representation=z_representation,
        )
        np.testing.assert_allclose(result, expected)

        # Numpy array input
        gate = ffsim.qiskit.DiagCoulombEvolutionJW(
            norb,
            np.stack((mat_aa, mat_ab, mat_bb)),
            time,
            z_representation=z_representation,
        )
        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )
        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )
        expected = ffsim.apply_diag_coulomb_evolution(
            small_vec,
            (mat_aa, mat_ab, mat_bb),
            time,
            norb=norb,
            nelec=nelec,
            z_representation=z_representation,
        )
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nocc(range(5)))
def test_random_diag_coulomb_mat_spinless(norb: int, nelec: int):
    """Test random spinless diag Coulomb gate gives correct output state."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        time = rng.uniform(-10, 10)

        # mat
        gate = ffsim.qiskit.DiagCoulombEvolutionSpinlessJW(norb, mat, time)
        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
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
def test_inverse_spinful(norb: int, nelec: tuple[int, int], z_representation: bool):
    """Test inverse."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        mat_aa = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        mat_ab = rng.standard_normal((norb, norb))
        mat_bb = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        time = rng.uniform(-10, 10)

        # mat
        gate = ffsim.qiskit.DiagCoulombEvolutionJW(
            norb, mat_aa, time, z_representation=z_representation
        )
        vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            ffsim.random.random_state_vector(dim, seed=rng), norb=norb, nelec=nelec
        )
        statevec = Statevector(vec).evolve(gate)
        statevec = statevec.evolve(gate.inverse())
        np.testing.assert_allclose(np.array(statevec), vec)

        # (mat_aa, mat_ab, mat_bb)
        gate = ffsim.qiskit.DiagCoulombEvolutionJW(
            norb,
            (mat_aa, mat_ab, mat_bb),
            time,
            z_representation=z_representation,
        )
        vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            ffsim.random.random_state_vector(dim, seed=rng), norb=norb, nelec=nelec
        )
        statevec = Statevector(vec).evolve(gate)
        statevec = statevec.evolve(gate.inverse())
        np.testing.assert_allclose(np.array(statevec), vec)

        # (mat_aa, None, mat_bb)
        gate = ffsim.qiskit.DiagCoulombEvolutionJW(
            norb,
            (mat_aa, None, mat_bb),
            time,
            z_representation=z_representation,
        )
        vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            ffsim.random.random_state_vector(dim, seed=rng), norb=norb, nelec=nelec
        )
        statevec = Statevector(vec).evolve(gate)
        statevec = statevec.evolve(gate.inverse())
        np.testing.assert_allclose(np.array(statevec), vec)
