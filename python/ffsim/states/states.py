# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fermionic quantum states."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import scipy.linalg
from pyscf.fci import cistring
from pyscf.fci.spin_op import contract_ss

from ffsim.gates.orbital_rotation import apply_orbital_rotation
from ffsim.spin import Spin


def dims(norb: int, nelec: tuple[int, int]) -> tuple[int, int]:
    """Get the dimensions of the FCI space.

    Args:
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.

    Returns:
        A pair of integers (dim_a, dim_b) representing the dimensions of the
        alpha- and beta- FCI space.
    """
    n_alpha, n_beta = nelec
    dim_a = math.comb(norb, n_alpha)
    dim_b = math.comb(norb, n_beta)
    return dim_a, dim_b


def dim(norb: int, nelec: tuple[int, int]) -> int:
    """Get the dimension of the FCI space.

    Args:
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.

    Returns:
        The dimension of the FCI space.
    """
    dim_a, dim_b = dims(norb, nelec)
    return dim_a * dim_b


def one_hot(shape: int | tuple[int, ...], index, *, dtype=complex):
    """Return an array of all zeros except for a one at a specified index.

    Args:
        shape: The desired shape of the array.
        index: The index at which to place a one.

    Returns:
        The one-hot vector.
    """
    vec = np.zeros(shape, dtype=dtype)
    vec[index] = 1
    return vec


def slater_determinant(
    norb: int,
    occupied_orbitals: tuple[Sequence[int], Sequence[int]],
    orbital_rotation: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    r"""Return a Slater determinant.

    A Slater determinant is a state of the form

    .. math::

        \mathcal{U} \lvert x \rangle,

    where :math:`\mathcal{U}` is an
    :doc:`orbital rotation </explanations/orbital-rotation>` and
    :math:`\lvert x \rangle` is an electronic configuration.

    Args:
        norb: The number of spatial orbitals.
        occupied_orbitals: The occupied orbitals in the electonic configuration.
            This is a pair of lists of integers, where the first list specifies the
            spin alpha orbitals and the second list specifies the spin beta
            orbitals.
        orbital_rotation: The optional orbital rotation. You can pass either a
            single Numpy array specifying the orbital rotation to apply to both
            spin sectors, or you can pass a pair of Numpy arrays to specify
            independent orbital rotations for spin alpha and spin beta.

    Returns:
        The Slater determinant as a statevector.
    """
    if norb == 0:
        return np.ones(1, dtype=complex)

    alpha_orbitals, beta_orbitals = occupied_orbitals
    n_alpha = len(alpha_orbitals)
    n_beta = len(beta_orbitals)
    nelec = (n_alpha, n_beta)
    dim1, dim2 = dims(norb, nelec)
    alpha_bits = np.zeros(norb, dtype=bool)
    alpha_bits[list(alpha_orbitals)] = 1
    alpha_string = int("".join("1" if b else "0" for b in alpha_bits[::-1]), base=2)
    alpha_index = cistring.str2addr(norb, n_alpha, alpha_string)
    beta_bits = np.zeros(norb, dtype=bool)
    beta_bits[list(beta_orbitals)] = 1
    beta_string = int("".join("1" if b else "0" for b in beta_bits[::-1]), base=2)
    beta_index = cistring.str2addr(norb, n_beta, beta_string)
    vec = one_hot((dim1, dim2), (alpha_index, beta_index), dtype=complex).reshape(-1)
    if orbital_rotation is not None:
        if isinstance(orbital_rotation, np.ndarray):
            vec = apply_orbital_rotation(
                vec, orbital_rotation, norb=norb, nelec=nelec, copy=False
            )
        else:
            orbital_rotation_a, orbital_rotation_b = orbital_rotation
            vec = apply_orbital_rotation(
                vec,
                orbital_rotation_a,
                norb=norb,
                nelec=nelec,
                spin=Spin.ALPHA,
                copy=False,
            )
            vec = apply_orbital_rotation(
                vec,
                orbital_rotation_b,
                norb=norb,
                nelec=nelec,
                spin=Spin.BETA,
                copy=False,
            )
    return vec


def hartree_fock_state(norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """Return the Hartree-Fock state.

    Args:
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.

    Returns:
        The Hartree-Fock state as a statevector.
    """
    n_alpha, n_beta = nelec
    return slater_determinant(
        norb=norb, occupied_orbitals=(range(n_alpha), range(n_beta))
    )


def slater_determinant_rdm(
    norb: int,
    occupied_orbitals: tuple[Sequence[int], Sequence[int]],
    orbital_rotation: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
    rank: int = 1,
    spin_summed: bool = True,
) -> np.ndarray:
    """Return the reduced density matrix of a `Slater determinant`_.

    Note:
        Currently, only rank 1 is supported.

    Args:
        norb: The number of spatial orbitals.
        occupied_orbitals: The occupied orbitals in the electonic configuration.
            This is a pair of lists of integers, where the first list specifies the
            spin alpha orbitals and the second list specifies the spin beta
            orbitals.
        orbital_rotation: The optional orbital rotation. You can pass either a
            single Numpy array specifying the orbital rotation to apply to both
            spin sectors, or you can pass a pair of Numpy arrays to specify
            independent orbital rotations for spin alpha and spin beta.
        rank: The rank of the reduced density matrix. I.e., rank 1 corresponds to the
            one-particle RDM, rank 2 corresponds to the 2-particle RDM, etc.
        spin_summed: Whether to sum over the spin index.

    Returns:
        The reduced density matrix of the Slater determinant.

    .. _Slater determinant: ffsim.html#ffsim.slater_determinant
    """
    if rank == 1:
        rdm_a = np.zeros((norb, norb), dtype=complex)
        rdm_b = np.zeros((norb, norb), dtype=complex)
        alpha_orbitals = np.array(occupied_orbitals[0])
        beta_orbitals = np.array(occupied_orbitals[1])
        if len(alpha_orbitals):
            rdm_a[(alpha_orbitals, alpha_orbitals)] = 1
        if len(beta_orbitals):
            rdm_b[(beta_orbitals, beta_orbitals)] = 1
        if orbital_rotation is not None:
            if isinstance(orbital_rotation, np.ndarray):
                orbital_rotation_a = orbital_rotation
                orbital_rotation_b = orbital_rotation
            else:
                orbital_rotation_a, orbital_rotation_b = orbital_rotation
            rdm_a = orbital_rotation_a.conj() @ rdm_a @ orbital_rotation_a.T
            rdm_b = orbital_rotation_b.conj() @ rdm_b @ orbital_rotation_b.T
        if spin_summed:
            return rdm_a + rdm_b
        return scipy.linalg.block_diag(rdm_a, rdm_b)
    raise NotImplementedError(
        f"Returning the rank {rank} reduced density matrix is currently not supported."
    )


def indices_to_strings(
    indices: Sequence[int], norb: int, nelec: tuple[int, int]
) -> list[str]:
    """Convert statevector indices to bitstrings.

    Example:

    .. code::

        import ffsim

        norb = 3
        nelec = (2, 1)
        dim = ffsim.dim(norb, nelec)
        ffsim.indices_to_strings(range(dim), norb, nelec)
        # output:
        # ['001011',
        #  '010011',
        #  '100011',
        #  '001101',
        #  '010101',
        #  '100101',
        #  '001110',
        #  '010110',
        #  '100110']
    """
    n_alpha, n_beta = nelec
    dim_b = math.comb(norb, n_beta)
    indices_a, indices_b = np.divmod(indices, dim_b)
    strings_a = cistring.addrs2str(norb=norb, nelec=n_alpha, addrs=indices_a)
    strings_b = cistring.addrs2str(norb=norb, nelec=n_beta, addrs=indices_b)
    return [
        f"{string_b:0{norb}b}{string_a:0{norb}b}"
        for string_a, string_b in zip(strings_a, strings_b)
    ]


# source: pyscf.fci.spin_op.spin_square0
# modified to support complex wavefunction
def spin_square(fcivec: np.ndarray, norb: int, nelec: tuple[int, int]):
    """Expectation value of spin squared operator on a state vector."""
    if np.issubdtype(fcivec.dtype, np.complexfloating):
        ci1 = contract_ss(fcivec.real, norb, nelec).astype(complex)
        ci1 += 1j * contract_ss(fcivec.imag, norb, nelec)
    else:
        ci1 = contract_ss(fcivec, norb, nelec)
    return np.einsum("ij,ij->", fcivec.reshape(ci1.shape), ci1.conj()).real
