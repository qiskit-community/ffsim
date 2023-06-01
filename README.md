# ffsim

Faster simulations of fermionic quantum circuits.

## What is ffsim?

ffsim is a high-performance simulator for fermionic quantum circuits that preserve particle number
and the Z component of spin. This category includes many quantum circuits used for quantum chemistry simulations.

## Features

- Fast simulation of fermionic quantum circuits that preserve particle number and the Z component of spin.
  ffsim supports the simulation of basic fermionic gates and includes specialized routines for simulation
  of molecular Hamiltonians in the "double-factorized" representation.
- Quantum computing software framework-agnostic.
  ffsim is programmed in a largely functional style using functions that take Numpy arrays as input and yield
  Numpy arrays as output. As a result, it can readily be used as a base for higher-level simulation frameworks.
- Compatible with PySCF. State vectors use the same indexing convention as PySCF's `fci` module.

## Limitations

- There is no support for operations that do not preserve particle number and the Z component of spin.

## Installation

In the future, binary wheels will be made available on PyPI. For now, you can install from source.

### Installing from source

Installing ffsim from source requires a Rust compiler to be present on the system.
A Rust compiler can be installed by following the instructions [here](https://www.rust-lang.org/tools/install).
Once the Rust compiler is installed, ffsim can be installed by running the command

    pip install .

from the root directory of the code repository. To install in editable mode, do

    pip install -e .
