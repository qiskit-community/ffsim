# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.linalg
from pyscf.fci import cistring
from pyscf.fci.direct_spin1 import (
    make_rdm1,
    make_rdm1s,
    make_rdm12,
    make_rdm12s,
    trans_rdm1,
    trans_rdm1s,
    trans_rdm12,
    trans_rdm12s,
)
from scipy.special import comb

from ffsim.cistring import gen_linkstr_index
from ffsim.gates.orbital_rotation import apply_orbital_rotation


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
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)
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


def one_hot(shape: tuple[int, ...], index, *, dtype=complex):
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
    orbital_rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Return a Slater determinant.

    Args:
        norb: The number of spatial orbitals.
        occupied_orbitals: A pair of lists of integers. The first list specifies
            the occupied alpha orbitals and the second list specifies the occupied
            beta orbitals.
        orbital_rotation: An optional orbital rotation to apply to the
            electron configuration. In other words, this is a unitary matrix that
            describes the orbitals of the Slater determinant.

    Returns:
        The Slater determinant as a statevector.
    """
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
        vec = apply_orbital_rotation(vec, orbital_rotation, norb=norb, nelec=nelec)
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
    orbital_rotation: np.ndarray | None = None,
    rank: int = 1,
    spin_summed: bool = True,
) -> np.ndarray:
    """Return the reduced density matrix of a Slater determinant.

    Note:
        Currently, only rank 1 is supported.

    Args:
        norb: The number of spatial orbitals.
        occupied_orbitals: A tuple of two sequences of integers. The first
            sequence contains the indices of the occupied alpha orbitals, and
            the second sequence similarly for the beta orbitals.
        orbital_rotation: An optional orbital rotation to apply to the
            electron configuration. In other words, this is a unitary matrix that
            describes the orbitals of the Slater determinant.
        rank: The rank of the reduced density matrix.
        spin_summed: Whether to sum over the spin index.

    Returns:
        The reduced density matrix of the Slater determinant.
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
            rdm_a = orbital_rotation.conj() @ rdm_a @ orbital_rotation.T
            rdm_b = orbital_rotation.conj() @ rdm_b @ orbital_rotation.T
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
        >>> norb = 3
        >>> nelec = (2, 1)
        >>> dim = ffsim.dim(norb, nelec)
        >>> ffsim.indices_to_strings(range(dim), norb, nelec)
        ['011001', '011010', '011100', '101001', '101010', '101100', '110001', \
'110010', '110100']
    """
    n_alpha, n_beta = nelec
    dim_b = comb(norb, n_beta, exact=True)
    indices_a, indices_b = np.divmod(indices, dim_b)
    strings_a = cistring.addrs2str(norb=norb, nelec=n_alpha, addrs=indices_a)
    strings_b = cistring.addrs2str(norb=norb, nelec=n_beta, addrs=indices_b)
    return [
        f"{string_a:0{norb}b}{string_b:0{norb}b}"
        for string_a, string_b in zip(strings_a, strings_b)
    ]


def rdm(
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    rank: int = 1,
    spin_summed: bool = True,
    reordered: bool = True,
    # TODO make this default to True
    # TODO document this argument
    return_lower_ranks: bool = False,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Return the reduced density matrix (RDM) of a state vector.

    The rank 1 RDM is defined as follows:

    .. code::

        rdm1[p, q] = ⟨p+ q⟩

    The definition of higher-rank RDMs depends on the ``reordered`` argument, which
    defaults to True.

    **reordered = True**

    The reordered RDMs are defined as follows:

    .. code::

        rdm2[p, q, r, s] = ⟨p+ r+ s q⟩
        rdm3[p, q, r, s, t, u] = ⟨p+ r+ t+ u s q⟩
        rdm4[p, q, r, s, t, u, v, w] = ⟨p+ r+ t+ v+ w u s q⟩

    **reordered = False**

    If reordered is set to False, the RDMs are defined as follows:

    .. code::

        rdm2[p, q, r, s] = ⟨p+ q r+ s⟩
        rdm3[p, q, r, s, t, u] = ⟨p+ q r+ s t+ u⟩
        rdm4[p, q, r, s, t, u, v, w] = ⟨p+ q r+ s t+ u v+ w⟩

    Note:
        Currently, only ranks 1 and 2 are supported.

    Args:
        vec: The state vector whose reduced density matrix is desired.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        rank: The rank of the reduced density matrix.
        spin_summed: Whether to sum over the spin index.
        reordered: Whether to reorder the indices of the reduced density matrix.

    Returns:
        The reduced density matrix.
    """
    n_alpha, n_beta = nelec
    link_index_a = gen_linkstr_index(range(norb), n_alpha)
    link_index_b = gen_linkstr_index(range(norb), n_beta)
    link_index = (link_index_a, link_index_b)
    vec_real = np.real(vec)
    vec_imag = np.imag(vec)
    if rank == 1:
        if spin_summed:
            return _rdm1_spin_summed(vec_real, vec_imag, norb, nelec, link_index)
        else:
            return _rdm1(vec_real, vec_imag, norb, nelec, link_index)
    if rank == 2:
        if spin_summed:
            return _rdm2_spin_summed(
                vec_real,
                vec_imag,
                norb,
                nelec,
                reordered,
                link_index,
                return_lower_ranks,
            )
        else:
            return _rdm2(
                vec_real,
                vec_imag,
                norb,
                nelec,
                reordered,
                link_index,
                return_lower_ranks,
            )
    raise NotImplementedError(
        f"Computing the rank {rank} reduced density matrix is currently not supported."
    )


def _rdm1_spin_summed(
    vec_real: np.ndarray,
    vec_imag: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    link_index: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    result = make_rdm1(vec_real, norb, nelec, link_index=link_index).astype(complex)
    result += make_rdm1(vec_imag, norb, nelec, link_index=link_index)
    # the rdm1 convention from pyscf is transposed
    result -= 1j * trans_rdm1(vec_real, vec_imag, norb, nelec, link_index=link_index)
    result += 1j * trans_rdm1(vec_imag, vec_real, norb, nelec, link_index=link_index)
    return result


def _rdm1(
    vec_real: np.ndarray,
    vec_imag: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    link_index: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    result = np.stack(make_rdm1s(vec_real, norb, nelec, link_index=link_index)).astype(
        complex
    )
    result += np.stack(make_rdm1s(vec_imag, norb, nelec, link_index=link_index))
    # the rdm1 convention from pyscf is transposed
    result -= 1j * np.stack(
        trans_rdm1s(vec_real, vec_imag, norb, nelec, link_index=link_index)
    )
    result += 1j * np.stack(
        trans_rdm1s(vec_imag, vec_real, norb, nelec, link_index=link_index)
    )
    return scipy.linalg.block_diag(*result)


def _rdm2_spin_summed(
    vec_real: np.ndarray,
    vec_imag: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    reordered: bool,
    link_index: tuple[np.ndarray, np.ndarray] | None,
    return_lower_ranks: bool,
) -> np.ndarray:
    rdm1_real, rdm2_real = make_rdm12(
        vec_real, norb, nelec, reorder=reordered, link_index=link_index
    )
    rdm1_imag, rdm2_imag = make_rdm12(
        vec_imag, norb, nelec, reorder=reordered, link_index=link_index
    )
    trans_rdm1_real_imag, trans_rdm2_real_imag = trans_rdm12(
        vec_real, vec_imag, norb, nelec, reorder=reordered, link_index=link_index
    )
    trans_rdm1_imag_real, trans_rdm2_imag_real = trans_rdm12(
        vec_imag, vec_real, norb, nelec, reorder=reordered, link_index=link_index
    )
    rdm2 = rdm2_real + rdm2_imag + 1j * (trans_rdm2_real_imag - trans_rdm2_imag_real)
    if not return_lower_ranks:
        return rdm2
    # use minus sign for 1j because the rdm1 convention from pyscf is transposed
    rdm1 = rdm1_real + rdm1_imag - 1j * (trans_rdm1_real_imag - trans_rdm1_imag_real)
    return rdm1, rdm2


def _rdm2(
    vec_real: np.ndarray,
    vec_imag: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    reordered: bool,
    link_index: tuple[np.ndarray, np.ndarray] | None,
    return_lower_ranks: bool,
) -> np.ndarray:
    rdms1_real, rdms2_real = make_rdm12s(
        vec_real, norb, nelec, reorder=reordered, link_index=link_index
    )
    rdms1_imag, rdms2_imag = make_rdm12s(
        vec_imag, norb, nelec, reorder=reordered, link_index=link_index
    )
    trans_rdms1_real_imag, trans_rdms2_real_imag = trans_rdm12s(
        vec_real, vec_imag, norb, nelec, reorder=reordered, link_index=link_index
    )
    trans_rdms1_imag_real, trans_rdms2_imag_real = trans_rdm12s(
        vec_imag, vec_real, norb, nelec, reorder=reordered, link_index=link_index
    )

    rdms2 = np.stack(rdms2_real).astype(complex)
    rdms2 += np.stack(rdms2_imag)
    # rdms2 is currently [rdm_aa, rdm_ab, rdm_bb]
    # the following line transforms it into [rdm_aa, rdm_ab, rdm_ba, rdm_bb]
    rdms2 = np.insert(rdms2, 2, rdms2[1].transpose(2, 3, 0, 1), axis=0)
    rdms2 += 1j * np.stack(trans_rdms2_real_imag)
    rdms2 -= 1j * np.stack(trans_rdms2_imag_real)
    rdm2 = np.zeros((2 * norb, 2 * norb, 2 * norb, 2 * norb), dtype=complex)
    rdm_aa, rdm_ab, rdm_ba, rdm_bb = rdms2
    rdm2[:norb, :norb, :norb, :norb] = rdm_aa
    rdm2[:norb, :norb, norb:, norb:] = rdm_ab
    rdm2[norb:, norb:, :norb, :norb] = rdm_ba
    rdm2[norb:, norb:, norb:, norb:] = rdm_bb
    if not return_lower_ranks:
        return rdm2

    rdms1 = np.stack(rdms1_real).astype(complex)
    rdms1 += np.stack(rdms1_imag)
    rdms1 -= 1j * np.stack(trans_rdms1_real_imag)
    rdms1 += 1j * np.stack(trans_rdms1_imag_real)
    rdm1 = scipy.linalg.block_diag(*rdms1)
    return rdm1, rdm2
