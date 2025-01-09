# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import numpy as np

import ffsim


def test_trace():
    norb = 50
    nelec = (3, 3)
    rng = np.random.default_rng(12345)
    ham = ffsim.random.random_diagonal_coulomb_hamiltonian(norb, real=True, seed=rng)
    t1 = ffsim.trace(ffsim.fermion_operator(ham), norb=norb, nelec=nelec)
    t2 = ffsim.trace(ham, norb=norb, nelec=nelec)
    np.testing.assert_allclose(t1, t2)
