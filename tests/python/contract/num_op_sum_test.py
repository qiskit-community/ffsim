# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for num op sum contraction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest

import ffsim


@pytest.mark.parametrize("norb", [4, 5])
def test_contract_num_op_sum(norb: int):
    """Test contracting sum of number operators."""
    rng = np.random.default_rng()
    for _ in range(50):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        nelec = (n_alpha, n_beta)
        alpha_orbitals = cast(Sequence[int], rng.choice(norb, n_alpha, replace=False))
        beta_orbitals = cast(Sequence[int], rng.choice(norb, n_beta, replace=False))
        occupied_orbitals = (alpha_orbitals, beta_orbitals)
        state = ffsim.slater_determinant(norb, occupied_orbitals)

        coeffs = rng.standard_normal(norb)
        result = ffsim.contract.contract_num_op_sum(
            state, coeffs, norb=norb, nelec=nelec
        )

        eig = 0
        for i in range(norb):
            for sigma in range(2):
                if i in occupied_orbitals[sigma]:
                    eig += coeffs[i]
        expected = eig * state

        np.testing.assert_allclose(result, expected)


def test_num_op_sum_to_linop():
    """Test converting a num op sum to a linear operator."""
    norb = 5
    rng = np.random.default_rng()
    n_alpha = rng.integers(1, norb + 1)
    n_beta = rng.integers(1, norb + 1)
    nelec = (n_alpha, n_beta)
    dim = ffsim.dim(norb, nelec)

    coeffs = rng.standard_normal(norb)
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    vec = ffsim.random.random_state_vector(dim, seed=rng)

    linop = ffsim.contract.num_op_sum_linop(
        coeffs, norb=norb, nelec=nelec, orbital_rotation=orbital_rotation
    )
    result = linop @ vec

    expected = ffsim.apply_orbital_rotation(
        vec, orbital_rotation.T.conj(), norb=norb, nelec=nelec
    )
    expected = ffsim.contract.contract_num_op_sum(
        expected, coeffs, norb=norb, nelec=nelec
    )
    expected = ffsim.apply_orbital_rotation(
        expected, orbital_rotation, norb=norb, nelec=nelec
    )

    np.testing.assert_allclose(result, expected)
