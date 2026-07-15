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

RNG = np.random.default_rng(42513779153765741156481720722198397462)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_random_ucj_op_spin_unbalanced(norb: int, nelec: tuple[int, int]):
    """Test random spin-unbalanced UCJ gate gives correct output state."""
    n_reps = 3
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        ucj_op = ffsim.random.random_ucj_op_spin_unbalanced(
            norb, n_reps=n_reps, with_final_orbital_rotation=True, seed=RNG
        )
        gate = ffsim.qiskit.UCJOpSpinUnbalancedJW(ucj_op)

        small_vec = ffsim.random.random_state_vector(dim, seed=RNG)
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
    n_reps = 3
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        ucj_op = ffsim.random.random_ucj_op_spin_balanced(
            norb, n_reps=n_reps, with_final_orbital_rotation=True, seed=RNG
        )
        gate = ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op)

        small_vec = ffsim.random.random_state_vector(dim, seed=RNG)
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
    norb = 4
    n_reps = 2
    generator = 1e-8j * ffsim.random.random_hermitian(norb, seed=RNG)
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


@pytest.mark.parametrize("norb", [4, 5])
def test_compressed_max_givens(norb: int):
    """Test that orb_rot_max_givens caps the number of XXPlusYY gates."""
    n_reps = 2
    generator = 0.1j * ffsim.random.random_hermitian(norb, seed=RNG)
    orbital_rotation = scipy.linalg.expm(generator)
    max_givens = norb * (norb - 1) // 4

    # Two orbital rotations per repetition, plus one final orbital rotation.
    n_orb_rots = 2 * n_reps + 1

    # Spin-balanced: each orbital rotation is applied to both spin sectors.
    ucj_op = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=np.zeros((n_reps, 2, norb, norb)),
        orbital_rotations=np.tile(orbital_rotation, (n_reps, 1, 1)),
        final_orbital_rotation=orbital_rotation,
    )
    gate = ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op, orb_rot_max_givens=max_givens)
    assert (
        gate.definition.decompose(reps=1).count_ops()["xx_plus_yy"]
        == 2 * n_orb_rots * max_givens
    )

    # Spin-unbalanced: independent alpha/beta rotations, each applied to both sectors.
    ucj_op_unbalanced = ffsim.UCJOpSpinUnbalanced(
        diag_coulomb_mats=np.zeros((n_reps, 3, norb, norb)),
        orbital_rotations=np.tile(orbital_rotation, (n_reps, 2, 1, 1)),
        final_orbital_rotation=np.tile(orbital_rotation, (2, 1, 1)),
    )
    gate = ffsim.qiskit.UCJOpSpinUnbalancedJW(
        ucj_op_unbalanced, orb_rot_max_givens=max_givens
    )
    assert (
        gate.definition.decompose(reps=1).count_ops()["xx_plus_yy"]
        == 2 * n_orb_rots * max_givens
    )

    # Spinless: one XXPlusYY gate per retained Givens rotation.
    ucj_op_spinless = ffsim.UCJOpSpinless(
        diag_coulomb_mats=np.zeros((n_reps, norb, norb)),
        orbital_rotations=np.tile(orbital_rotation, (n_reps, 1, 1)),
        final_orbital_rotation=orbital_rotation,
    )
    gate = ffsim.qiskit.UCJOpSpinlessJW(ucj_op_spinless, orb_rot_max_givens=max_givens)
    assert (
        gate.definition.decompose(reps=1).count_ops()["xx_plus_yy"]
        == n_orb_rots * max_givens
    )


@pytest.mark.parametrize("norb", [4, 5])
def test_compressed_max_layers(norb: int):
    """Test that orb_rot_max_layers caps the number of XXPlusYY gates."""
    n_reps = 2
    generator = 0.1j * ffsim.random.random_hermitian(norb, seed=RNG)
    orbital_rotation = scipy.linalg.expm(generator)
    max_layers = norb // 2

    # Two orbital rotations per repetition, plus one final orbital rotation.
    n_orb_rots = 2 * n_reps + 1

    # The number of retained Givens rotations matches the compressed decomposition.
    givens_rotations, _ = ffsim.linalg.givens_decomposition(
        orbital_rotation, max_layers=max_layers
    )
    n_expected = len(givens_rotations)

    # Spin-balanced: each orbital rotation is applied to both spin sectors.
    ucj_op = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=np.zeros((n_reps, 2, norb, norb)),
        orbital_rotations=np.tile(orbital_rotation, (n_reps, 1, 1)),
        final_orbital_rotation=orbital_rotation,
    )
    gate = ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op, orb_rot_max_layers=max_layers)
    assert (
        gate.definition.decompose(reps=1).count_ops()["xx_plus_yy"]
        == 2 * n_orb_rots * n_expected
    )

    # Spin-unbalanced: independent alpha/beta rotations, each applied to both sectors.
    ucj_op_unbalanced = ffsim.UCJOpSpinUnbalanced(
        diag_coulomb_mats=np.zeros((n_reps, 3, norb, norb)),
        orbital_rotations=np.tile(orbital_rotation, (n_reps, 2, 1, 1)),
        final_orbital_rotation=np.tile(orbital_rotation, (2, 1, 1)),
    )
    gate = ffsim.qiskit.UCJOpSpinUnbalancedJW(
        ucj_op_unbalanced, orb_rot_max_layers=max_layers
    )
    assert (
        gate.definition.decompose(reps=1).count_ops()["xx_plus_yy"]
        == 2 * n_orb_rots * n_expected
    )

    # Spinless: one XXPlusYY gate per retained Givens rotation.
    ucj_op_spinless = ffsim.UCJOpSpinless(
        diag_coulomb_mats=np.zeros((n_reps, norb, norb)),
        orbital_rotations=np.tile(orbital_rotation, (n_reps, 1, 1)),
        final_orbital_rotation=orbital_rotation,
    )
    gate = ffsim.qiskit.UCJOpSpinlessJW(ucj_op_spinless, orb_rot_max_layers=max_layers)
    assert (
        gate.definition.decompose(reps=1).count_ops()["xx_plus_yy"]
        == n_orb_rots * n_expected
    )


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nocc(exhaustive=False)
)
def test_random_ucj_op_spinless(norb: int, nelec: int):
    """Test random spin-balanced UCJ gate gives correct output state."""
    n_reps = 3
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        ucj_op = ffsim.random.random_ucj_op_spinless(
            norb, n_reps=n_reps, with_final_orbital_rotation=True, seed=RNG
        )
        gate = ffsim.qiskit.UCJOpSpinlessJW(ucj_op)

        small_vec = ffsim.random.random_state_vector(dim, seed=RNG)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )

        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.apply_unitary(small_vec, ucj_op, norb=norb, nelec=nelec)

        np.testing.assert_allclose(result, expected)
