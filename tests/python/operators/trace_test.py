import numpy as np
import time
import ffsim

def test_trace(norb = 50, nelec = (3,3)):
    n = norb
    
    h = np.matrix(np.random.rand(n,n))
    h = h + h.H
    V = np.array([np.matrix(np.random.rand(n,n)),np.matrix(np.random.rand(n,n))])
    H = ffsim.DiagonalCoulombHamiltonian(h,V)
    
    tic = time.time()
    t1 = ffsim.trace(ffsim.fermion_operator(H),norb=norb, nelec=nelec)
    toc = time.time()
    print(f"Trace = {t1}. Took {toc-tic}s by converting DiagonalCoulombHamiltonian to FermionOperator")
    
    tic = time.time()
    t2 = ffsim.trace(H, norb=n, nelec=nelec)
    toc = time.time()
    print(f"Trace = {t2}. Took {toc-tic}s using DiagonalCoulombHamiltonian._trace()")

    assert np.abs(t1-t2)<1e-3, "The two methods do not match!"