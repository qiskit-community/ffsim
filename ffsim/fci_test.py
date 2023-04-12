# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for FCI utils."""

from __future__ import annotations

import itertools

import numpy as np

from ffsim.fci import contract_diag_coulomb, contract_num_op_sum
from ffsim.random_utils import random_hermitian
from ffsim.states import slater_determinant


def test_contract_diag_coulomb():
    """Test contracting core tensor."""
    norb = 5
    rng = np.random.default_rng()
    n_alpha = rng.integers(1, norb + 1)
    n_beta = rng.integers(1, norb + 1)
    occupied_orbitals = (
        rng.choice(norb, n_alpha, replace=False),
        rng.choice(norb, n_beta, replace=False),
    )
    nelec = tuple(len(orbs) for orbs in occupied_orbitals)
    state = slater_determinant(norb, occupied_orbitals)

    core_tensor = np.real(np.array(random_hermitian(norb, seed=rng)))
    result = contract_diag_coulomb(core_tensor, state, nelec)

    eig = 0
    for i, j in itertools.product(range(norb), repeat=2):
        for sigma, tau in itertools.product(range(2), repeat=2):
            if i in occupied_orbitals[sigma] and j in occupied_orbitals[tau]:
                eig += 0.5 * core_tensor[i, j]
    expected = eig * state

    np.testing.assert_allclose(result, expected, atol=1e-8)


def test_contract_num_op_sum():
    """Test contracting sum of number operators."""
    norb = 5
    rng = np.random.default_rng()
    n_alpha = rng.integers(1, norb + 1)
    n_beta = rng.integers(1, norb + 1)
    occupied_orbitals = (
        rng.choice(norb, n_alpha, replace=False),
        rng.choice(norb, n_beta, replace=False),
    )
    nelec = tuple(len(orbs) for orbs in occupied_orbitals)
    state = slater_determinant(norb, occupied_orbitals)

    coeffs = rng.standard_normal(norb)
    result = contract_num_op_sum(coeffs, state, nelec)

    eig = 0
    for i in range(norb):
        for sigma in range(2):
            if i in occupied_orbitals[sigma]:
                eig += coeffs[i]
    expected = eig * state

    np.testing.assert_allclose(result, expected, atol=1e-8)
