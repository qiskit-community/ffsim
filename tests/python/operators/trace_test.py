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
import time
import ffsim

def test_trace():
    norb = 50
    nelec = (3,3)

    rng = np.random.default_rng(12345)
    h = np.matrix(rng.random((norb,norb)))
    h = h + h.H
    V = np.array([np.matrix(rng.random((norb,norb))),np.matrix(rng.random((norb,norb)))])
    H = ffsim.DiagonalCoulombHamiltonian(h,V)
    
    tic = time.time()
    t1 = ffsim.trace(ffsim.fermion_operator(H),norb=norb, nelec=nelec)
    toc = time.time()
    print(f"Trace = {t1}. Took {toc-tic}s by converting DiagonalCoulombHamiltonian to FermionOperator")
    
    tic = time.time()
    t2 = ffsim.trace(H, norb=norb, nelec=nelec)
    toc = time.time()
    print(f"Trace = {t2}. Took {toc-tic}s using DiagonalCoulombHamiltonian._trace()")

    assert np.abs(t1-t2)<1e-3, "The two methods do not match!"