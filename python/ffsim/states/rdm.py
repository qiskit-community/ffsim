# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions to compute reduced density matrices."""

from __future__ import annotations

import numpy as np
import scipy.linalg
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
from typing_extensions import deprecated

from ffsim.cistring import gen_linkstr_index


@deprecated("ffsim.rdm is deprecated. Instead, use ffsim.rdms.")
def rdm(
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    rank: int = 1,
    spin_summed: bool = True,
    reordered: bool = True,
    return_lower_ranks: bool = True,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Return the reduced density matrix (RDM) or matrices of a state vector.

    .. warning::
        This function is deprecated. Use :func:`ffsim.rdms` instead.

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
        return_lower_ranks: Whether to return lower rank RDMs in addition to the
            specified rank. If True, then this function returns all RDMs up to and
            including the specified rank, in increasing order of rank. For example,
            if `rank=2` then a tuple `(rdm1, rdm2)` is returned.

    Returns:
        The reduced density matrix or matrices. If `return_lower_ranks` is False,
        then a single matrix is returned. If `return_lower_ranks` is True, then
        a `rank`-length tuple of matrices is returned, containing the RDMs up to
        the specified rank in increasing order of rank.
    """
    n_alpha, n_beta = nelec
    link_index_a = gen_linkstr_index(range(norb), n_alpha)
    link_index_b = gen_linkstr_index(range(norb), n_beta)
    link_index = (link_index_a, link_index_b)
    if rank == 1:
        if spin_summed:
            return _rdm1_spin_summed(vec, norb, nelec, link_index)
        else:
            return _rdm1(vec, norb, nelec, link_index)
    if rank == 2:
        if spin_summed:
            return _rdm2_spin_summed_deprecated(
                vec, norb, nelec, reordered, link_index, return_lower_ranks
            )
        else:
            return _rdm2(vec, norb, nelec, reordered, link_index, return_lower_ranks)
    raise NotImplementedError(
        f"Computing the rank {rank} reduced density matrix is currently not supported."
    )


def _rdm2_spin_summed_deprecated(
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    reordered: bool,
    link_index: tuple[np.ndarray, np.ndarray] | None,
    return_lower_ranks: bool,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    rdm1_real, rdm2_real = make_rdm12(
        vec.real, norb, nelec, reorder=reordered, link_index=link_index
    )
    rdm1_imag, rdm2_imag = make_rdm12(
        vec.imag, norb, nelec, reorder=reordered, link_index=link_index
    )
    trans_rdm1_real_imag, trans_rdm2_real_imag = trans_rdm12(
        vec.real, vec.imag, norb, nelec, reorder=reordered, link_index=link_index
    )
    rdm2 = _assemble_rdm2_spin_summed(rdm2_real, rdm2_imag, trans_rdm2_real_imag)
    if not return_lower_ranks:
        return rdm2
    rdm1 = _assemble_rdm1_spin_summed(rdm1_real, rdm1_imag, trans_rdm1_real_imag)
    return rdm1, rdm2


def _rdm1(
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    link_index: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    rdms1_real = make_rdm1s(vec.real, norb, nelec, link_index=link_index)
    rdms1_imag = make_rdm1s(vec.imag, norb, nelec, link_index=link_index)
    trans_rdms1_real_imag = trans_rdm1s(
        vec.real, vec.imag, norb, nelec, link_index=link_index
    )
    trans_rdms1_imag_real = trans_rdm1s(
        vec.imag, vec.real, norb, nelec, link_index=link_index
    )
    return _assemble_rdm1(
        rdms1_real, rdms1_imag, trans_rdms1_real_imag, trans_rdms1_imag_real
    )


def _assemble_rdm1(
    rdms1_real: tuple[np.ndarray, np.ndarray],
    rdms1_imag: tuple[np.ndarray, np.ndarray],
    trans_rdms1_real_imag: tuple[np.ndarray, np.ndarray],
    trans_rdms1_imag_real: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    rdms1 = np.stack(rdms1_real).astype(complex)
    rdms1 += np.stack(rdms1_imag)
    # use minus sign for real-imag and plus sign for imag-real because
    # the rdm1 convention in pyscf is transposed
    rdms1 -= 1j * np.stack(trans_rdms1_real_imag)
    rdms1 += 1j * np.stack(trans_rdms1_imag_real)
    return scipy.linalg.block_diag(*rdms1)


def _rdm2(
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    reordered: bool,
    link_index: tuple[np.ndarray, np.ndarray] | None,
    return_lower_ranks: bool,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    rdms1_real, rdms2_real = make_rdm12s(
        vec.real, norb, nelec, reorder=reordered, link_index=link_index
    )
    rdms1_imag, rdms2_imag = make_rdm12s(
        vec.imag, norb, nelec, reorder=reordered, link_index=link_index
    )
    trans_rdms1_real_imag, trans_rdms2_real_imag = trans_rdm12s(
        vec.real, vec.imag, norb, nelec, reorder=reordered, link_index=link_index
    )
    trans_rdms1_imag_real, trans_rdms2_imag_real = trans_rdm12s(
        vec.imag, vec.real, norb, nelec, reorder=reordered, link_index=link_index
    )
    rdm2 = _assemble_rdm2(
        rdms2_real, rdms2_imag, trans_rdms2_real_imag, trans_rdms2_imag_real
    )
    if not return_lower_ranks:
        return rdm2
    rdm1 = _assemble_rdm1(
        rdms1_real, rdms1_imag, trans_rdms1_real_imag, trans_rdms1_imag_real
    )
    return rdm1, rdm2


def _assemble_rdm2(
    rdms2_real: tuple[np.ndarray, np.ndarray, np.ndarray],
    rdms2_imag: tuple[np.ndarray, np.ndarray, np.ndarray],
    trans_rdms2_real_imag: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    trans_rdms2_imag_real: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    rdms2 = np.stack(rdms2_real).astype(complex)
    rdms2 += np.stack(rdms2_imag)
    # rdms2 is currently [rdm_aa, rdm_ab, rdm_bb]
    # the following line transforms it into [rdm_aa, rdm_ab, rdm_ba, rdm_bb]
    rdms2 = np.insert(rdms2, 2, rdms2[1].transpose(2, 3, 0, 1), axis=0)
    rdms2 += 1j * np.stack(trans_rdms2_real_imag)
    rdms2 -= 1j * np.stack(trans_rdms2_imag_real)

    norb, _, _, _ = rdms2_real[0].shape
    rdm2 = np.zeros((2 * norb, 2 * norb, 2 * norb, 2 * norb), dtype=complex)
    rdm_aa, rdm_ab, rdm_ba, rdm_bb = rdms2
    rdm2[:norb, :norb, :norb, :norb] = rdm_aa
    rdm2[:norb, :norb, norb:, norb:] = rdm_ab
    rdm2[norb:, norb:, :norb, :norb] = rdm_ba
    rdm2[norb:, norb:, norb:, norb:] = rdm_bb
    return rdm2


def rdms(
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    rank: int = 1,
    spin_summed: bool = False,
    reorder: bool = True,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Return the spin-separated reduced density matrices of a state vector.

    The rank 1 RDM is defined as follows:

    .. code::

        rdm1[p, q] = ⟨p+ q⟩

    The definition of higher-rank RDMs depends on the ``reorder`` argument, which
    defaults to True.

    **reorder = True**

    The reordered RDMs are defined as follows:

    .. code::

        rdm2[p, q, r, s] = ⟨p+ r+ s q⟩
        rdm3[p, q, r, s, t, u] = ⟨p+ r+ t+ u s q⟩
        rdm4[p, q, r, s, t, u, v, w] = ⟨p+ r+ t+ v+ w u s q⟩

    **reorder = False**

    If reorder is set to False, the RDMs are defined as follows:

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
        spin_summed: Whether to return the "spin-summed" RDMs.
        reorder: Whether to reorder the indices of the reduced density matrix.

    Returns:
        The reduced density matrices.
        All RDMs up to and including the specified rank are returned, in increasing
        order of rank. For example, if `rank=2` then a tuple `(rdm1, rdm2)` is returned.
        The 1-RDMs are: (alpha-alpha, beta-beta).
        The spin-summed 1-RDM is alpha-alpha + alpha-beta.
        The 2-RDMs are: (alpha-alpha, alpha-beta, beta-beta).
        The spin-summed 2-RDM is alpha-alpha + alpha-beta + beta-alpha + beta-beta.
    """
    n_alpha, n_beta = nelec
    link_index_a = gen_linkstr_index(range(norb), n_alpha)
    link_index_b = gen_linkstr_index(range(norb), n_beta)
    link_index = (link_index_a, link_index_b)
    if rank == 1:
        if spin_summed:
            return _rdm1_spin_summed(vec, norb, nelec, link_index)
        else:
            return _rdm1s(vec, norb, nelec, link_index)
    if rank == 2:
        if spin_summed:
            return _rdm2_spin_summed(vec, norb, nelec, reorder, link_index)
        else:
            return _rdm2s(vec, norb, nelec, reorder, link_index)
    raise NotImplementedError(
        f"Computing the rank {rank} reduced density matrix is currently not supported."
    )


def _rdm1s(
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    link_index: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    rdm1s_real = make_rdm1s(vec.real, norb, nelec, link_index=link_index)
    rdm1s_imag = make_rdm1s(vec.imag, norb, nelec, link_index=link_index)
    trans_rdm1s_real_imag = trans_rdm1s(
        vec.real, vec.imag, norb, nelec, link_index=link_index
    )
    return _assemble_rdm1s(rdm1s_real, rdm1s_imag, trans_rdm1s_real_imag)


def _assemble_rdm1s(
    rdm1s_real: tuple[np.ndarray, np.ndarray],
    rdm1s_imag: tuple[np.ndarray, np.ndarray],
    trans_rdm1s_real_imag: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    rdm1s = np.stack(rdm1s_real).astype(complex)
    rdm1s += np.stack(rdm1s_imag)
    # use minus sign for real-imag and plus sign for imag-real because
    # the rdm1 convention in pyscf is transposed
    rdm1s -= 1j * np.stack(trans_rdm1s_real_imag)
    rdm1s += 1j * np.stack(trans_rdm1s_real_imag).transpose(0, 2, 1)
    return rdm1s


def _rdm2s(
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    reorder: bool,
    link_index: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    rdm1s_real, rdm2s_real = make_rdm12s(
        vec.real, norb, nelec, reorder=reorder, link_index=link_index
    )
    rdm1s_imag, rdm2s_imag = make_rdm12s(
        vec.imag, norb, nelec, reorder=reorder, link_index=link_index
    )
    trans_rdm1s_real_imag, trans_rdm2s_real_imag = trans_rdm12s(
        vec.real, vec.imag, norb, nelec, reorder=reorder, link_index=link_index
    )
    rdm1 = _assemble_rdm1s(rdm1s_real, rdm1s_imag, trans_rdm1s_real_imag)
    rdm2 = _assemble_rdm2s(rdm2s_real, rdm2s_imag, trans_rdm2s_real_imag)
    return rdm1, rdm2


def _assemble_rdm2s(
    rdm2s_real: tuple[np.ndarray, np.ndarray, np.ndarray],
    rdm2s_imag: tuple[np.ndarray, np.ndarray, np.ndarray],
    trans_rdm2s_real_imag: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    rdm2s = np.stack(rdm2s_real).astype(complex)
    rdm2s += np.stack(rdm2s_imag)
    aa, ab, _, bb = trans_rdm2s_real_imag
    rdm2s += 1j * np.stack([aa, ab, bb])
    rdm2s -= 1j * np.stack(
        [
            aa.transpose(3, 2, 1, 0),
            ab.transpose(1, 0, 3, 2),
            bb.transpose(3, 2, 1, 0),
        ]
    )
    return rdm2s


def _rdm1_spin_summed(
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    link_index: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    rdm1_real = make_rdm1(vec.real, norb, nelec, link_index=link_index)
    rdm1_imag = make_rdm1(vec.imag, norb, nelec, link_index=link_index)
    trans_rdm1_real_imag = trans_rdm1(
        vec.real, vec.imag, norb, nelec, link_index=link_index
    )
    return _assemble_rdm1_spin_summed(rdm1_real, rdm1_imag, trans_rdm1_real_imag)


def _assemble_rdm1_spin_summed(
    rdm1_real: np.ndarray,
    rdm1_imag: np.ndarray,
    trans_rdm1_real_imag: np.ndarray,
) -> np.ndarray:
    # use minus sign for 1j because the rdm1 convention from pyscf is transposed
    return rdm1_real + rdm1_imag - 1j * (trans_rdm1_real_imag - trans_rdm1_real_imag.T)


def _rdm2_spin_summed(
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    reorder: bool,
    link_index: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    rdm1_real, rdm2_real = make_rdm12(
        vec.real, norb, nelec, reorder=reorder, link_index=link_index
    )
    rdm1_imag, rdm2_imag = make_rdm12(
        vec.imag, norb, nelec, reorder=reorder, link_index=link_index
    )
    trans_rdm1_real_imag, trans_rdm2_real_imag = trans_rdm12(
        vec.real, vec.imag, norb, nelec, reorder=reorder, link_index=link_index
    )
    rdm1 = _assemble_rdm1_spin_summed(rdm1_real, rdm1_imag, trans_rdm1_real_imag)
    rdm2 = _assemble_rdm2_spin_summed(rdm2_real, rdm2_imag, trans_rdm2_real_imag)
    return rdm1, rdm2


def _assemble_rdm2_spin_summed(
    rdm2_real: np.ndarray,
    rdm2_imag: np.ndarray,
    trans_rdm2_real_imag: np.ndarray,
) -> np.ndarray:
    return (
        rdm2_real
        + rdm2_imag
        + 1j * (trans_rdm2_real_imag - trans_rdm2_real_imag.transpose(3, 2, 1, 0))
    )
