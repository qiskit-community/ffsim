=================================
Welcome to ffsim's documentation!
=================================

ffsim is a software library for simulating fermionic quantum circuits that preserve particle number
and the Z component of spin. This category includes many quantum circuits used for quantum chemistry simulations.
By exploiting the symmetries and using specialized algorithms, ffsim can simulate these circuits much faster
than a generic quantum circuit simulator.


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

ffsim module
------------
The following objects are exposed at the top-level ``ffsim.*`` namespace.

.. toctree::
   :maxdepth: 2

   api/ffsim

Submodules
----------
The following submodules are available. Many of the objects from these submodules are already exposed in the top-level ``ffsim.*`` namespace.

.. toctree::
   :maxdepth: 1

   api/ffsim.contract
   api/ffsim.gates
   api/ffsim.hamiltonians
   api/ffsim.linalg
   api/ffsim.protocols
   api/ffsim.random
   api/ffsim.states
   api/ffsim.testing
   api/ffsim.trotter
   api/ffsim.variational
