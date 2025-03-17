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


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_random(norb: int, nelec: tuple[int, int]):
    """Test on random fermion Hamiltonian."""
    rng = np.random.default_rng(4482)
    op = ffsim.random.random_fermion_hamiltonian(norb, seed=rng)
    linop = ffsim.linear_operator(op, norb=norb, nelec=nelec)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=rng)
    expected_result = ffsim.qiskit.ffsim_vec_to_qiskit_vec(linop @ vec, norb, nelec)

    qubit_op = ffsim.qiskit.jordan_wigner(op)
    qubit_op_sparse = qubit_op.to_matrix(sparse=True)
    actual_result = qubit_op_sparse @ ffsim.qiskit.ffsim_vec_to_qiskit_vec(
        vec, norb, nelec
    )
    np.testing.assert_allclose(actual_result, expected_result, atol=1e-12)

    qubit_op = ffsim.qiskit.jordan_wigner(op, norb=norb)
    qubit_op_sparse = qubit_op.to_matrix(sparse=True)
    actual_result = qubit_op_sparse @ ffsim.qiskit.ffsim_vec_to_qiskit_vec(
        vec, norb, nelec
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
    rng = np.random.default_rng(7431)
    norb_x = 2
    norb_y = 2
    norb = norb_x * norb_y
    nelec = (norb // 2, norb // 2)
    op = ffsim.fermi_hubbard_2d(
        norb_x=norb_x,
        norb_y=norb_y,
        tunneling=rng.uniform(-10, 10),
        interaction=rng.uniform(-10, 10),
        chemical_potential=rng.uniform(-10, 10),
        nearest_neighbor_interaction=rng.uniform(-10, 10),
        periodic=False,
    )
    linop = ffsim.linear_operator(op, norb=norb, nelec=nelec)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=rng)
    qubit_op = ffsim.qiskit.jordan_wigner(op)
    qubit_op_sparse = qubit_op.to_matrix(sparse=True)
    actual_result = qubit_op_sparse @ ffsim.qiskit.ffsim_vec_to_qiskit_vec(
        vec, norb, nelec
    )
    expected_result = ffsim.qiskit.ffsim_vec_to_qiskit_vec(linop @ vec, norb, nelec)
    np.testing.assert_allclose(actual_result, expected_result)
