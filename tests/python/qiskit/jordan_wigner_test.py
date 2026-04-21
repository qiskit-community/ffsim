# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Jordan-Wigner transformation."""

import numpy as np
import pytest

import ffsim
import ffsim.random.random

RNG = np.random.default_rng(29474543787640228645686111044798504547)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_random(norb: int, nelec: tuple[int, int]):
    """Test on random fermion Hamiltonian."""
    op = ffsim.random.random_fermion_hamiltonian(norb, seed=RNG)
    linop = ffsim.linear_operator(op, norb=norb, nelec=nelec)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    expected_result = ffsim.qiskit.ffsim_vec_to_qiskit_vec(linop @ vec, norb, nelec)

    qubit_op = ffsim.qiskit.jordan_wigner(op, norb=norb)
    qubit_op_sparse = qubit_op.to_matrix(sparse=True)
    actual_result = qubit_op_sparse @ ffsim.qiskit.ffsim_vec_to_qiskit_vec(
        vec, norb, nelec
    )
    np.testing.assert_allclose(actual_result, expected_result, atol=1e-12)


@pytest.mark.parametrize("norb", range(5))
def test_inferred_norb(norb: int):
    """Test that norb is correctly inferred from the operator when not specified."""
    op = ffsim.random.random_fermion_hamiltonian(norb, seed=RNG)
    norb_in_op = (1 + max(orb for term in op for _, _, orb in term)) if op else 0
    nelec = (norb_in_op // 2, norb_in_op - norb_in_op // 2)
    linop = ffsim.linear_operator(op, norb=norb_in_op, nelec=nelec)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb_in_op, nelec), seed=RNG)
    expected_result = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
        linop @ vec, norb_in_op, nelec
    )

    qubit_op = ffsim.qiskit.jordan_wigner(op)
    assert qubit_op.num_qubits == 2 * norb_in_op
    qubit_op_sparse = qubit_op.to_matrix(sparse=True)
    actual_result = qubit_op_sparse @ ffsim.qiskit.ffsim_vec_to_qiskit_vec(
        vec, norb_in_op, nelec
    )
    np.testing.assert_allclose(actual_result, expected_result, atol=1e-12)


def test_more_norb():
    """Test specifying more spatial orbitals than are present in the operator."""
    op = ffsim.FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.cre_b(1)): 1.0,
            (ffsim.cre_a(2), ffsim.cre_b(2)): 1.0,
        }
    )
    qubit_op = ffsim.qiskit.jordan_wigner(op, norb=5)
    assert sorted(qubit_op.to_sparse_list()) == sorted(
        [
            ("YZZZZX", [1, 2, 3, 4, 5, 6], np.complex128(-0.25j)),
            ("YZZZZY", [1, 2, 3, 4, 5, 6], np.complex128(-0.25 + 0j)),
            ("XZZZZX", [1, 2, 3, 4, 5, 6], np.complex128(0.25 + 0j)),
            ("XZZZZY", [1, 2, 3, 4, 5, 6], np.complex128(-0.25j)),
            ("YZZZZX", [2, 3, 4, 5, 6, 7], np.complex128(-0.25j)),
            ("YZZZZY", [2, 3, 4, 5, 6, 7], np.complex128(-0.25 + 0j)),
            ("XZZZZX", [2, 3, 4, 5, 6, 7], np.complex128(0.25 + 0j)),
            ("XZZZZY", [2, 3, 4, 5, 6, 7], np.complex128(-0.25j)),
        ]
    )


def test_bad_norb():
    """Test passing bad number of spatial orbitals raises errors."""
    op = ffsim.FermionOperator({(ffsim.cre_a(3),): 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        _ = ffsim.qiskit.jordan_wigner(op, norb=-1)
    with pytest.raises(ValueError, match="fewer"):
        _ = ffsim.qiskit.jordan_wigner(op, norb=3)


def test_hubbard():
    """Test on Hubbard model"""
    norb_x = 2
    norb_y = 2
    norb = norb_x * norb_y
    nelec = (norb // 2, norb // 2)
    op = ffsim.fermi_hubbard_2d(
        norb_x=norb_x,
        norb_y=norb_y,
        tunneling=RNG.uniform(-10, 10),
        interaction=RNG.uniform(-10, 10),
        chemical_potential=RNG.uniform(-10, 10),
        nearest_neighbor_interaction=RNG.uniform(-10, 10),
        periodic=False,
    )
    linop = ffsim.linear_operator(op, norb=norb, nelec=nelec)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    qubit_op = ffsim.qiskit.jordan_wigner(op)
    qubit_op_sparse = qubit_op.to_matrix(sparse=True)
    actual_result = qubit_op_sparse @ ffsim.qiskit.ffsim_vec_to_qiskit_vec(
        vec, norb, nelec
    )
    expected_result = ffsim.qiskit.ffsim_vec_to_qiskit_vec(linop @ vec, norb, nelec)
    np.testing.assert_allclose(actual_result, expected_result)
