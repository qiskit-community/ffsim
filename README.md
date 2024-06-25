# ffsim

<!-- start introduction -->

ffsim is a software library for simulating fermionic quantum circuits that conserve particle number and the Z component of spin. This category includes many quantum circuits used for quantum chemistry simulations. By exploiting the symmetries and using specialized algorithms, ffsim can simulate these circuits much faster than a generic quantum circuit simulator.

<!-- end introduction -->

## Documentation

Documentation is located at the [project website](https://qiskit-community.github.io/ffsim/).

## Installation

<!-- start installation -->

We recommend installing ffsim using pip, when possible:

```bash
pip install ffsim
```

This method won't work natively on Windows, however. Refer to the [installation instructions](https://qiskit-community.github.io/ffsim/install.html) for information about using ffsim on Windows, as well as instructions for installing from source and running ffsim in a container.

<!-- end installation -->

## Code example

<!-- start code-example -->

```python
import numpy as np
import pyscf

import ffsim

# Build an N2 molecule
mol = pyscf.gto.Mole()
mol.build(atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]], basis="6-31g", symmetry="Dooh")

# Get molecular data
scf = pyscf.scf.RHF(mol).run()
mol_data = ffsim.MolecularData.from_scf(scf, active_space=range(4, mol.nao_nr()))
norb, nelec = mol_data.norb, mol_data.nelec

# Generate a random orbital rotation
orbital_rotation = ffsim.random.random_unitary(norb, seed=1234)

# Create the Hartree-Fock state and apply the orbital rotation to it
vec = ffsim.hartree_fock_state(norb, nelec)
vec = ffsim.apply_orbital_rotation(vec, orbital_rotation, norb=norb, nelec=nelec)

# Convert the Hamiltonian to a Scipy LinearOperator
linop = ffsim.linear_operator(mol_data.hamiltonian, norb=norb, nelec=nelec)

# Compute the energy of the state
energy = np.vdot(vec, linop @ vec).real
print(energy)  # prints -104.17181289596
```

<!-- end code-example -->

## Citing ffsim

<!-- start citing -->

You can cite ffsim using the following BibTeX:

```bibtex
@software{ffsim,
  author = {{The ffsim developers}},
  title = {{ffsim: Faster simulations of fermionic quantum circuits.}},
  url = {https://github.com/qiskit-community/ffsim}
}
```

<!-- end citing -->

## Developer guide

See the [developer guide](https://github.com/qiskit-community/ffsim/blob/main/CONTRIBUTING.md) for instructions on contributing code to ffsim.
