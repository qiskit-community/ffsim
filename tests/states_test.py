# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test states."""

import numpy as np

import ffsim


def test_slater_determinant():
    """Test Slater determinant."""
    norb = 5
    nelec = ffsim.testing.random_nelec(norb)
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec)
    occ_a, occ_b = occupied_orbitals

    one_body_tensor = ffsim.random.random_hermitian(norb)
    eigs, orbital_rotation = np.linalg.eigh(one_body_tensor)
    eig = sum(eigs[occ_a]) + sum(eigs[occ_b])
    state = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )

    hamiltonian = ffsim.contract.hamiltonian_linop(
        one_body_tensor=one_body_tensor, norb=norb, nelec=nelec
    )
    np.testing.assert_allclose(hamiltonian @ state, eig * state)
