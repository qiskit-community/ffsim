# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for unitary cluster Jastrow gate."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

import ffsim


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_random_ucj_op_spin_unbalanced(norb: int, nelec: tuple[int, int]):
    """Test random spin-unbalanced UCJ gate gives correct output state."""
    rng = np.random.default_rng()
    n_reps = 3
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        ucj_op = ffsim.random.random_ucj_op_spin_unbalanced(
            norb, n_reps=n_reps, with_final_orbital_rotation=True, seed=rng
        )
        gate = ffsim.qiskit.UCJOpSpinUnbalancedJW(ucj_op)

        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )

        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.apply_unitary(small_vec, ucj_op, norb=norb, nelec=nelec)

        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_random_ucj_op_spin_balanced(norb: int, nelec: tuple[int, int]):
    """Test random spin-balanced UCJ gate gives correct output state."""
    rng = np.random.default_rng()
    n_reps = 3
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        ucj_op = ffsim.random.random_ucj_op_spin_balanced(
            norb, n_reps=n_reps, with_final_orbital_rotation=True, seed=rng
        )
        gate = ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op)

        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )

        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.apply_unitary(small_vec, ucj_op, norb=norb, nelec=nelec)

        np.testing.assert_allclose(result, expected)


def test_ucj_op_tol():
    """Test passing tol to UCJ gates."""
    rng = np.random.default_rng()
    norb = 4
    n_reps = 2
    generator = 1e-8j * ffsim.random.random_hermitian(norb, seed=rng)
    orbital_rotation = scipy.linalg.expm(generator)

    # Spin-balanced
    ucj_op = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=np.zeros((n_reps, 2, norb, norb)),
        orbital_rotations=np.tile(orbital_rotation, (n_reps, 1, 1)),
    )
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op, tol=1e-7), qubits)
    assert "xx_plus_yy" not in circuit.decompose(reps=2).count_ops()

    # Spin-unbalanced
    ucj_op_unbalanced = ffsim.UCJOpSpinUnbalanced(
        diag_coulomb_mats=np.zeros((n_reps, 3, norb, norb)),
        orbital_rotations=np.tile(orbital_rotation, (n_reps, 2, 1, 1)),
    )
    circuit = QuantumCircuit(qubits)
    circuit.append(
        ffsim.qiskit.UCJOpSpinUnbalancedJW(ucj_op_unbalanced, tol=1e-7), qubits
    )
    assert "xx_plus_yy" not in circuit.decompose(reps=2).count_ops()

    # Spinless
    ucj_op_spinless = ffsim.UCJOpSpinless(
        diag_coulomb_mats=np.zeros((n_reps, norb, norb)),
        orbital_rotations=np.tile(orbital_rotation, (n_reps, 1, 1)),
    )
    qubits_spinless = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits_spinless)
    circuit.append(
        ffsim.qiskit.UCJOpSpinlessJW(ucj_op_spinless, tol=1e-7), qubits_spinless
    )
    assert "xx_plus_yy" not in circuit.decompose(reps=2).count_ops()


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nocc(exhaustive=False)
)
def test_random_ucj_op_spinless(norb: int, nelec: int):
    """Test random spin-balanced UCJ gate gives correct output state."""
    rng = np.random.default_rng()
    n_reps = 3
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        ucj_op = ffsim.random.random_ucj_op_spinless(
            norb, n_reps=n_reps, with_final_orbital_rotation=True, seed=rng
        )
        gate = ffsim.qiskit.UCJOpSpinlessJW(ucj_op)

        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )

        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.apply_unitary(small_vec, ucj_op, norb=norb, nelec=nelec)

        np.testing.assert_allclose(result, expected)
