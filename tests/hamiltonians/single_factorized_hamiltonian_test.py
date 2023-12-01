# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for single-factorized Hamiltonian."""

from __future__ import annotations

import numpy as np
import pytest

import ffsim


@pytest.mark.parametrize(
    "norb, nelec, cholesky",
    [
        (4, (2, 2), False),
        (4, (2, 2), True),
        (4, (2, 1), False),
        (4, (2, 1), True),
        (4, (2, 0), False),
        (4, (2, 0), True),
    ],
)
def test_linear_operator(norb: int, nelec: tuple[int, int], cholesky: bool):
    """Test linear operator."""
    rng = np.random.default_rng(2474)

    dim = ffsim.dim(norb, nelec)
    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    two_body_tensor = ffsim.random.random_two_body_tensor(norb, seed=rng, dtype=float)
    constant = rng.standard_normal()
    mol_hamiltonian = ffsim.MolecularHamiltonian(
        one_body_tensor, two_body_tensor, constant
    )

    df_hamiltonian = ffsim.SingleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_hamiltonian, cholesky=cholesky
    )

    actual_linop = ffsim.linear_operator(df_hamiltonian, norb, nelec)
    expected_linop = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    dim = ffsim.dim(norb, nelec)
    state = ffsim.random.random_statevector(dim, seed=rng)

    actual = actual_linop @ state
    expected = expected_linop @ state

    np.testing.assert_allclose(actual, expected)
