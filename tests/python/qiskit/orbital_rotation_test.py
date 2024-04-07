# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for orbital rotation gate."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

import ffsim


@pytest.mark.parametrize(
    "norb, nelec, spin", ffsim.testing.generate_norb_nelec_spin(range(5))
)
def test_random_orbital_rotation(norb: int, nelec: tuple[int, int], spin: ffsim.Spin):
    """Test random orbital rotation circuit gives correct output state."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        mat = ffsim.random.random_unitary(norb, seed=rng)
        gate = ffsim.qiskit.OrbitalRotationJW(mat, spin=spin)

        small_vec = ffsim.random.random_statevector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )

        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.apply_orbital_rotation(
            small_vec, mat, norb=norb, nelec=nelec, spin=spin
        )

        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "norb, nelec, spin", ffsim.testing.generate_norb_nelec_spin(range(5))
)
def test_inverse(norb: int, nelec: tuple[int, int], spin: ffsim.Spin):
    """Test inverse."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        mat = ffsim.random.random_unitary(norb, seed=rng)
        gate = ffsim.qiskit.OrbitalRotationJW(mat, spin=spin)

        vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            ffsim.random.random_statevector(dim, seed=rng), norb=norb, nelec=nelec
        )

        statevec = Statevector(vec).evolve(gate)
        statevec = statevec.evolve(gate.inverse())

        np.testing.assert_allclose(np.array(statevec), vec)
