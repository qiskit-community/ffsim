# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Wick's theorem utilities."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse.linalg

import ffsim
from ffsim.states.wick import expectation_one_body_power, expectation_one_body_product


@pytest.mark.parametrize(
    "norb, occupied_orbitals",
    [
        (4, ([0, 1], [0, 1])),
        (3, ([0], [1, 2])),
        (2, ([], [0])),
    ],
)
def test_expectation_product(norb: int, occupied_orbitals: tuple[list[int], list[int]]):
    """Test expectation product."""
    occ_a, occ_b = occupied_orbitals
    nelec = len(occ_a), len(occ_b)
    dim = ffsim.dim(norb, nelec)

    rng = np.random.default_rng()

    # generate random one-body tensors
    n_tensors = 6
    one_body_tensors = []
    linops = []
    for _ in range(n_tensors):
        one_body_tensor = rng.standard_normal((norb, norb)).astype(complex)
        one_body_tensor += 1j * rng.standard_normal((norb, norb))
        linop = ffsim.contract.one_body_linop(one_body_tensor, norb=norb, nelec=nelec)
        one_body_tensors.append(
            scipy.linalg.block_diag(one_body_tensor, one_body_tensor)
        )
        linops.append(linop)

    # generate a random Slater determinant
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    rdm = scipy.linalg.block_diag(
        *ffsim.slater_determinant_rdms(
            norb, occupied_orbitals, orbital_rotation=orbital_rotation
        )
    )

    # get the full statevector
    vec = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )

    product_op = scipy.sparse.linalg.LinearOperator(
        shape=(dim, dim), matvec=lambda x: x
    )
    for i in range(n_tensors):
        product_op = product_op @ linops[i]
        computed = expectation_one_body_product(rdm, one_body_tensors[: i + 1])
        target = np.vdot(vec, product_op @ vec)
        np.testing.assert_allclose(computed, target)


def test_expectation_power():
    """Test expectation power."""
    norb = 4
    occupied_orbitals = ([0, 2], [1, 3])

    occ_a, occ_b = occupied_orbitals
    nelec = len(occ_a), len(occ_b)
    dim = ffsim.dim(norb, nelec)

    rng = np.random.default_rng()

    # generate a random one-body tensor
    one_body_tensor = rng.standard_normal((norb, norb)).astype(complex)
    one_body_tensor += 1j * rng.standard_normal((norb, norb))
    linop = ffsim.contract.one_body_linop(one_body_tensor, norb=norb, nelec=nelec)
    one_body_tensor = scipy.linalg.block_diag(one_body_tensor, one_body_tensor)

    # generate a random Slater determinant
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    rdm = scipy.linalg.block_diag(
        *ffsim.slater_determinant_rdms(
            norb, occupied_orbitals, orbital_rotation=orbital_rotation
        )
    )

    # get the full statevector
    vec = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )

    powered_op = scipy.sparse.linalg.LinearOperator(
        shape=(dim, dim), matvec=lambda x: x
    )
    for power in range(4):
        computed = expectation_one_body_power(rdm, one_body_tensor, power)
        target = np.vdot(vec, powered_op @ vec)
        np.testing.assert_allclose(computed, target)
        powered_op = powered_op @ linop
