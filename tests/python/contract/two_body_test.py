# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for two-body operator contraction."""

from __future__ import annotations

import itertools

import numpy as np

import ffsim
from ffsim.variational.uccsd import uccsd_restricted_linear_operator

RNG = np.random.default_rng(99651117001088794077543029456876666776)


def one_body_operator(one_body_tensor: np.ndarray) -> ffsim.FermionOperator:
    """Return a FermionOperator representing a one-body operator."""
    norb, _ = one_body_tensor.shape
    op = ffsim.FermionOperator({})
    for p, q in itertools.product(range(norb), repeat=2):
        coeff = one_body_tensor[p, q]
        op += ffsim.FermionOperator(
            {
                (ffsim.cre_a(p), ffsim.des_a(q)): coeff,
                (ffsim.cre_b(p), ffsim.des_b(q)): coeff,
            }
        )
    return op


def two_body_operator(two_body_tensor: np.ndarray) -> ffsim.FermionOperator:
    """Return a FermionOperator representing a two-body operator."""
    norb, _, _, _ = two_body_tensor.shape
    op = ffsim.FermionOperator({})
    for p, q, r, s in itertools.product(range(norb), repeat=4):
        coeff = 0.5 * two_body_tensor[p, q, r, s]
        op += ffsim.FermionOperator(
            {
                (ffsim.cre_a(p), ffsim.cre_a(r), ffsim.des_a(s), ffsim.des_a(q)): coeff,
                (ffsim.cre_a(p), ffsim.cre_b(r), ffsim.des_b(s), ffsim.des_a(q)): coeff,
                (ffsim.cre_b(p), ffsim.cre_a(r), ffsim.des_a(s), ffsim.des_b(q)): coeff,
                (ffsim.cre_b(p), ffsim.cre_b(r), ffsim.des_b(s), ffsim.des_b(q)): coeff,
            }
        )
    return op


def test_two_body_linop_hermitian_real():
    """Test converting real hermitian two-body operator to a linear operator."""
    norb = 5
    nelec = (3, 2)

    # Generate random two-body tensor
    one_body = ffsim.random.random_real_symmetric_matrix(norb, seed=RNG)
    two_body = ffsim.random.random_two_body_tensor(norb, seed=RNG, dtype=float)

    # Get linear operator from contract
    linop_contract = ffsim.contract.two_body_linop(
        two_body, norb=norb, nelec=nelec, one_body_tensor=one_body
    )

    # Get linear operator from FermionOperator
    ferm_op = one_body_operator(one_body) + two_body_operator(two_body)
    linop_ferm = ffsim.linear_operator(ferm_op, norb=norb, nelec=nelec)

    # Generate random vector
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)

    # Test operator application
    result_contract = linop_contract @ vec
    result_ferm = linop_ferm @ vec
    np.testing.assert_allclose(result_contract, result_ferm)

    # Test adjoint operator application
    result_contract = linop_contract.adjoint() @ vec
    result_ferm = linop_ferm.adjoint() @ vec
    np.testing.assert_allclose(result_contract, result_ferm)


def test_two_body_linop_hermitian_complex():
    """Test converting hermitian two-body operator to a linear operator."""
    norb = 5
    nelec = (3, 2)

    # Generate random one- and two-body tensors
    one_body = ffsim.random.random_hermitian(norb, seed=RNG)
    two_body = ffsim.random.random_two_body_tensor(norb, seed=RNG)

    # Get linear operator from contract
    linop_contract = ffsim.contract.two_body_linop(
        two_body, norb=norb, nelec=nelec, one_body_tensor=one_body
    )

    # Get linear operator from FermionOperator
    ferm_op = one_body_operator(one_body) + two_body_operator(two_body)
    linop_ferm = ffsim.linear_operator(ferm_op, norb=norb, nelec=nelec)

    # Generate random vector
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)

    # Test operator application
    result_contract = linop_contract @ vec
    result_ferm = linop_ferm @ vec
    np.testing.assert_allclose(result_contract, result_ferm)

    # Test adjoint operator application
    result_contract = linop_contract.adjoint() @ vec
    result_ferm = linop_ferm.adjoint() @ vec
    np.testing.assert_allclose(result_contract, result_ferm)


def test_two_body_linop_antihermitian_real():
    """Test converting real antihermitian two-body operator to a linear operator."""
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)

    # Generate random UCCSD operator
    uccsd_op = ffsim.random.random_uccsd_op_restricted_real(norb, nocc, seed=RNG)

    # Get linear operator from contract
    linop_contract = uccsd_restricted_linear_operator(
        t1=uccsd_op.t1, t2=uccsd_op.t2, norb=norb, nelec=nelec
    )

    # Get linear operator from FermionOperator
    uccsd_gen = ffsim.uccsd_generator_restricted(t1=uccsd_op.t1, t2=uccsd_op.t2)
    linop_ferm = ffsim.linear_operator(uccsd_gen, norb=norb, nelec=nelec)

    # Generate random vector
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)

    # Test operator application
    result_contract = linop_contract @ vec
    result_ferm = linop_ferm @ vec
    np.testing.assert_allclose(result_contract, result_ferm)

    # Test adjoint operator application
    result_contract = linop_contract.adjoint() @ vec
    result_ferm = linop_ferm.adjoint() @ vec
    np.testing.assert_allclose(result_contract, result_ferm)
    np.testing.assert_allclose(result_contract, -linop_contract @ vec)


def test_two_body_linop_antihermitian_complex():
    """Test converting antihermitian two-body operator to a linear operator."""
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)

    # Generate random UCCSD operator
    uccsd_op = ffsim.random.random_uccsd_op_restricted(norb, nocc, seed=RNG)

    # Get linear operator from contract
    linop_contract = uccsd_restricted_linear_operator(
        t1=uccsd_op.t1, t2=uccsd_op.t2, norb=norb, nelec=nelec
    )

    # Get linear operator from FermionOperator
    uccsd_gen = ffsim.uccsd_generator_restricted(t1=uccsd_op.t1, t2=uccsd_op.t2)
    linop_ferm = ffsim.linear_operator(uccsd_gen, norb=norb, nelec=nelec)

    # Generate random vector
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)

    # Test operator application
    result_contract = linop_contract @ vec
    result_ferm = linop_ferm @ vec
    np.testing.assert_allclose(result_contract, result_ferm)

    # Test adjoint operator application
    result_contract = linop_contract.adjoint() @ vec
    result_ferm = linop_ferm.adjoint() @ vec
    np.testing.assert_allclose(result_contract, result_ferm)
    np.testing.assert_allclose(result_contract, -linop_contract @ vec)
