=================================
Welcome to ffsim's documentation!
=================================

ffsim is a software library for simulating fermionic quantum circuits that conserve particle number
and the Z component of spin. This category includes many quantum circuits used for quantum chemistry simulations.
By exploiting the symmetries and using specialized algorithms, ffsim can simulate these circuits much faster
than a generic quantum circuit simulator.

ffsim's source code is located at https://github.com/qiskit-community/ffsim.


Tutorials
=========

.. toctree::
   :maxdepth: 1

   tutorials/01-introduction
   tutorials/02-orbital-rotation
   tutorials/03-double-factorized
   tutorials/04-lucj
   tutorials/05-entanglement-forging
   tutorials/06-fermion-operator


API Reference
=============

.. toctree::
   :maxdepth: 2

   api/ffsim
   api/ffsim.contract
   api/ffsim.linalg
   api/ffsim.random
   api/ffsim.testing
