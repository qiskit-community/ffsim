"""Implement the fermionic backpropagation algorithm for UCJ energy calculation."""

from ffsim.hamiltonians.molecular_hamiltonian import MolecularHamiltonian
from ffsim.variational.ucj_spin_balanced import UCJOpSpinBalanced
from ffsim.variational.ucj_spin_unbalanced import UCJOpSpinUnbalanced 
from ffsim.variational.ucj_spinless import UCJOpSpinless

from typing import Sequence




def ucj_energy_spin_balanced(
    ucj_op: UCJOpSpinBalanced, 
    hamiltonian: MolecularHamiltonian, 
    nelec: tuple[int, int],
    *,
    occupied_orbitals: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> float:
    """Compute the UCJ energy for a spin-balanced system 
    using the fermionic backpropagation outlined in https://arxiv.org/abs/2607.21337.
    
    Args:
        hamiltonian: The Hamiltonian.
        ucj_op: The UCJ operator. Must have n_reps=1, with an optional final orbital rotation.
        nelec: The number of alpha and beta electrons.
        occupied_orbitals: The occupied orbitals for the reference state. Defaults to the 
            Hartree-Fock state.
    
    Returns: 
        The expectation value of the Hamiltonian with respect to the UCJ state. 
    """
    ...

def ucj_energy_spin_unbalanced(
    ucj_op: UCJOpSpinUnbalanced,
    hamiltonian: MolecularHamiltonian,
    nelec: tuple[int, int],
    *,
    occupied_orbitals: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> float:
    """Compute the UCJ energy for a spin-unbalanced system 
    using the fermionic backpropagation outlined in https://arxiv.org/abs/2607.21337.
    
    Args:
        hamiltonian: The Hamiltonian.
        ucj_op: The UCJ operator. Must have n_reps=1, with an optional final orbital rotation.
        nelec: The number of alpha and beta electrons.
        occupied_orbitals: The occupied orbitals for the reference state. Defaults to the 
            Hartree-Fock state.

    Returns:
        The expectation value of the Hamiltonian with respect to the UCJ state.
    """
    ...


def ucj_energy_spinless(
    ucj_op: UCJOpSpinless,
    hamiltonian: MolecularHamiltonian,
    nelec: int,
    *,
    occupied_orbitals: Sequence[Sequence[int]],
) -> float:
    """Compute the UCJ energy for a spinless system 
    using the fermionic backpropagation outlined in https://arxiv.org/abs/2607.21337.
    
    Args:
        hamiltonian: The Hamiltonian.
        ucj_op: The UCJ operator. Must have n_reps=1, with an optional final orbital rotation.
        nelec: The number of electrons.
        occupied_orbitals: The occupied orbitals for the reference state. Defaults to the 
            Hartree-Fock state.

    Returns:
        The expectation value of the Hamiltonian with respect to the UCJ state.
    """
    ...