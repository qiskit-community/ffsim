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


@pytest.mark.parametrize("norb, spin", ffsim.testing.generate_norb_spin(range(5)))
def test_random_orbital_coeffs(norb: int, spin: ffsim.Spin):
    """Test circuit with crandom orbital coefficients gives correct output state."""
    rng = np.random.default_rng()
    nocc = norb // 2
    nelec = (
        nocc if spin & ffsim.Spin.ALPHA else 0,
        nocc if spin & ffsim.Spin.BETA else 0,
    )
    for _ in range(3):
        orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
        orbital_coeffs = orbital_rotation[:, :nocc]
        gate = ffsim.qiskit.PrepareSlaterDeterminantJW(orbital_coeffs, spin=spin)

        statevec = Statevector.from_int(0, 2 ** (2 * norb)).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        occupied_orbitals = (
            range(nocc) if spin & ffsim.Spin.ALPHA else [],
            range(nocc) if spin & ffsim.Spin.BETA else [],
        )
        expected = ffsim.slater_determinant(
            norb, occupied_orbitals, orbital_rotation=orbital_rotation
        )

        ffsim.testing.assert_allclose_up_to_global_phase(result, expected)
