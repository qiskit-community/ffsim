# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Qiskit utilities."""

import numpy as np

import ffsim


def test_ffsim_to_qiskit_roundtrip():
    """Test converting statevector between ffsim and Qiskit gives consistent results."""
    norb = 5
    nelec = 3, 2
    big_dim = 2 ** (2 * norb)
    small_dim = ffsim.dim(norb, nelec)
    rng = np.random.default_rng(9940)
    ffsim_vec = ffsim.random.random_state_vector(small_dim, seed=rng)
    qiskit_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(ffsim_vec, norb=norb, nelec=nelec)
    assert qiskit_vec.shape == (big_dim,)
    ffsim_vec_again = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
        qiskit_vec, norb=norb, nelec=nelec
    )
    assert ffsim_vec_again.shape == (small_dim,)
    np.testing.assert_array_equal(ffsim_vec, ffsim_vec_again)
