# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Givens rotation ansatz gate."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

import ffsim


def brickwork(norb: int, n_layers: int):
    for i in range(n_layers):
        for j in range(i % 2, norb - 1, 2):
            yield (j, j + 1)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_random_givens_ansatz_operator_spinful(norb: int, nelec: tuple[int, int]):
    """Test random Givens rotation ansatz gives correct output state."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        n_layers = 2 * norb
        interaction_pairs = list(brickwork(norb, n_layers))
        thetas = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
        givens_ansatz_op = ffsim.GivensAnsatzOperator(norb, interaction_pairs, thetas)
        gate = ffsim.qiskit.GivensAnsatzOperatorJW(givens_ansatz_op)

        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )

        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.apply_unitary(
            small_vec, givens_ansatz_op, norb=norb, nelec=nelec
        )

        np.testing.assert_allclose(result, expected)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("norb, nocc", ffsim.testing.generate_norb_nocc(range(5)))
def test_random_givens_ansatz_operator_spinless(norb: int, nocc: int):
    """Test random spinless Givens rotation ansatz gives correct output state."""
    rng = np.random.default_rng()
    nelec = (nocc, 0)
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        n_layers = 2 * norb
        interaction_pairs = list(brickwork(norb, n_layers))
        thetas = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
        givens_ansatz_op = ffsim.GivensAnsatzOperator(norb, interaction_pairs, thetas)
        gate = ffsim.qiskit.GivensAnsatzOperatorSpinlessJW(givens_ansatz_op)

        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )

        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.apply_unitary(
            small_vec, givens_ansatz_op, norb=norb, nelec=nelec
        )

        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_random_spinful(norb: int, nelec: tuple[int, int]):
    """Test random Givens rotation ansatz gives correct output state."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        interaction_pairs = list(brickwork(norb, norb))
        thetas = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
        phis = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
        phase_angles = rng.uniform(-np.pi, np.pi, size=norb)

        givens_ansatz_op = ffsim.GivensAnsatzOp(
            norb, interaction_pairs, thetas=thetas, phis=phis, phase_angles=phase_angles
        )
        gate = ffsim.qiskit.GivensAnsatzOpJW(givens_ansatz_op)

        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )

        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.apply_unitary(
            small_vec, givens_ansatz_op, norb=norb, nelec=nelec
        )

        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nocc(range(5)))
def test_random_spinless(norb: int, nelec: int):
    """Test random spinless Givens rotation ansatz gives correct output state."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        interaction_pairs = list(brickwork(norb, norb))
        thetas = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
        phis = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
        phase_angles = rng.uniform(-np.pi, np.pi, size=norb)

        givens_ansatz_op = ffsim.GivensAnsatzOp(
            norb, interaction_pairs, thetas=thetas, phis=phis, phase_angles=phase_angles
        )
        gate = ffsim.qiskit.GivensAnsatzOpSpinlessJW(givens_ansatz_op)
        assert gate.num_qubits == norb

        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )

        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.apply_unitary(
            small_vec, givens_ansatz_op, norb=norb, nelec=nelec
        )

        np.testing.assert_allclose(result, expected)
