# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Slater determinant preparation gate."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

import ffsim


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_prepare_hartree_fock_jw(norb: int, nelec: tuple[int, int]):
    """Test preparing Hartree-Fock state."""
    gate = ffsim.qiskit.PrepareHartreeFockJW(norb, nelec)

    statevec = Statevector.from_int(0, 2 ** (2 * norb)).evolve(gate)
    result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
        np.array(statevec), norb=norb, nelec=nelec
    )

    expected = ffsim.hartree_fock_state(norb, nelec)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_random_slater_determinant(norb: int, nelec: tuple[int, int]):
    """Test random Slater determinant circuit gives correct output state."""
    rng = np.random.default_rng()
    for _ in range(3):
        occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec)
        orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
        gate = ffsim.qiskit.PrepareSlaterDeterminantJW(
            norb, occupied_orbitals, orbital_rotation=orbital_rotation
        )

        statevec = Statevector.from_int(0, 2 ** (2 * norb)).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.slater_determinant(
            norb, occupied_orbitals, orbital_rotation=orbital_rotation
        )

        ffsim.testing.assert_allclose_up_to_global_phase(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_slater_determinant_no_rotation(norb: int, nelec: tuple[int, int]):
    """Test Slater determinant circuit with no orbital rotation."""
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec)

    gate = ffsim.qiskit.PrepareSlaterDeterminantJW(norb, occupied_orbitals)

    statevec = Statevector.from_int(0, 2 ** (2 * norb)).evolve(gate)
    result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
        np.array(statevec), norb=norb, nelec=nelec
    )

    expected = ffsim.slater_determinant(norb, occupied_orbitals)

    ffsim.testing.assert_allclose_up_to_global_phase(result, expected)
