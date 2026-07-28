"""Implement the fermionic backpropagation algorithm for UCJ energy calculation."""

from ffsim.hamiltonians.molecular_hamiltonian import MolecularHamiltonian
from ffsim.variational.ucj_spin_balanced import UCJOpSpinBalanced
from ffsim.variational.ucj_spin_unbalanced import UCJOpSpinUnbalanced 
from ffsim.variational.ucj_spinless import UCJOpSpinless
from ffsim.variational.util import validate_interaction_pairs 

from typing import Sequence

def ucj_energy_spin_balanced(
    ucj_op: UCJOpSpinBalanced, 
    hamiltonian: MolecularHamiltonian, 
    nelec: tuple[int, int],
    interaction_pairs: tuple[
        list[tuple[int, int]] | None, list[tuple[int, int]] | None
    ]
    | None = None,
    *,
    occupied_orbitals: tuple[Sequence[int], Sequence[int]] | None = None,
) -> float:
    """Compute the UCJ energy for a spin-balanced system 
    using the fermionic backpropagation outlined in https://arxiv.org/abs/2607.21337.
    
    Args:
        hamiltonian: The Hamiltonian.
        ucj_op: The UCJ operator. Must have n_reps=1, with an optional final orbital rotation.
        nelec: The number of alpha and beta electrons.
        interaction_pairs: The interaction pairs to consider. If None, all pairs are considered.
        occupied_orbitals: The occupied orbitals for the reference state. Defaults to the 
            Hartree-Fock state.
    
    Returns: 
        The expectation value of the Hamiltonian with respect to the UCJ state. 
    """

    pairs_aa, pairs_ab = interaction_pairs if interaction_pairs is not None else (None, None) 
    validate_interaction_pairs(pairs_aa, ordered=False) 
    validate_interaction_pairs(pairs_ab, ordered=True)
    ...


def ucj_energy_spin_unbalanced(
    ucj_op: UCJOpSpinUnbalanced,
    hamiltonian: MolecularHamiltonian,
    nelec: tuple[int, int],
    interaction_pairs: tuple[
        list[tuple[int, int]] | None, 
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ]
    | None = None,
    *,
    occupied_orbitals: tuple[Sequence[int], Sequence[int]] | None = None,
) -> float:
    """Compute the UCJ energy for a spin-unbalanced system 
    using the fermionic backpropagation outlined in https://arxiv.org/abs/2607.21337.
    
    Args:
        hamiltonian: The Hamiltonian.
        ucj_op: The UCJ operator. Must have n_reps=1, with an optional final orbital rotation.
        nelec: The number of alpha and beta electrons.
        interaction_pairs: The interaction pairs to consider. If None, all pairs are considered.
        occupied_orbitals: The occupied orbitals for the reference state. Defaults to the 
            Hartree-Fock state.

    Returns:
        The expectation value of the Hamiltonian with respect to the UCJ state.
    """
    pairs_aa, pairs_ab, pairs_bb = interaction_pairs if interaction_pairs is not None else (None, None, None)
    validate_interaction_pairs(pairs_aa, ordered=False)
    validate_interaction_pairs(pairs_ab, ordered=True)
    validate_interaction_pairs(pairs_bb, ordered=False)
    ...


def ucj_energy_spinless(
    ucj_op: UCJOpSpinless,
    hamiltonian: MolecularHamiltonian,
    nelec: int,
    *,
    occupied_orbitals: Sequence[int] | None = None,
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