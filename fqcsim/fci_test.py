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

import itertools

import numpy as np

from fqcsim.fci import (
    contract_core_tensor,
    contract_num_op_sum,
)
from fqcsim.random_utils import (
    random_hermitian,
)
from fqcsim.states import slater_determinant


def test_contract_core_tensor():
    """Test contracting core tensor."""
    n_orbitals = 5
    rng = np.random.default_rng()
    n_alpha = rng.integers(1, n_orbitals + 1)
    n_beta = rng.integers(1, n_orbitals + 1)
    occupied_orbitals = (
        rng.choice(n_orbitals, n_alpha, replace=False),
        rng.choice(n_orbitals, n_beta, replace=False),
    )
    n_electrons = tuple(len(orbs) for orbs in occupied_orbitals)
    state = slater_determinant(n_orbitals, occupied_orbitals)

    core_tensor = np.real(np.array(random_hermitian(n_orbitals, seed=rng)))
    result = contract_core_tensor(core_tensor, state, n_electrons)

    eig = 0
    for i, j in itertools.product(range(n_orbitals), repeat=2):
        for sigma, tau in itertools.product(range(2), repeat=2):
            if i in occupied_orbitals[sigma] and j in occupied_orbitals[tau]:
                eig += 0.5 * core_tensor[i, j]
    expected = eig * state

    np.testing.assert_allclose(result, expected, atol=1e-8)


def test_contract_num_op_sum():
    """Test contracting sum of number operators."""
    n_orbitals = 5
    rng = np.random.default_rng()
    n_alpha = rng.integers(1, n_orbitals + 1)
    n_beta = rng.integers(1, n_orbitals + 1)
    occupied_orbitals = (
        rng.choice(n_orbitals, n_alpha, replace=False),
        rng.choice(n_orbitals, n_beta, replace=False),
    )
    n_electrons = tuple(len(orbs) for orbs in occupied_orbitals)
    state = slater_determinant(n_orbitals, occupied_orbitals)

    coeffs = rng.standard_normal(n_orbitals)
    result = contract_num_op_sum(coeffs, state, n_electrons)

    eig = 0
    for i in range(n_orbitals):
        for sigma in range(2):
            if i in occupied_orbitals[sigma]:
                eig += coeffs[i]
    expected = eig * state

    np.testing.assert_allclose(result, expected, atol=1e-8)
