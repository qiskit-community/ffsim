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

import dataclasses
from collections.abc import Iterator
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from opt_einsum import contract
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
from pyscf.fci.rdm import make_dm123, make_dm1234, reorder_dm123, reorder_dm1234

from ffsim._cistring import gen_linkstr_index
from ffsim.linalg.util import rotate_one_body_tensor, rotate_two_body_tensor
from ffsim.states.states import spinful_to_spinless_vec

if TYPE_CHECKING:
    from ffsim.hamiltonians.molecular_hamiltonian import MolecularHamiltonian


@dataclasses.dataclass
class ReducedDensityMatrix:
    """Class to store one- and two-body reduced density matrices."""

    one_rdm: np.ndarray
    """The one-body reduced density matrix."""
    two_rdm: np.ndarray
    """The two-body reduced density matrix."""

    def rotated(self, orbital_rotation: np.ndarray) -> ReducedDensityMatrix:
        r"""Apply an orbital rotation to the reduced density matrices.

        Args:
            orbital_rotation: The orbital rotation.

        Returns:
            The rotated reduced density matrices.
        """
        # An RDM rotates by the opposite convention from a Hamiltonian tensor,
        # so pass the conjugate
        rotation = orbital_rotation.conj()
        return ReducedDensityMatrix(
            one_rdm=rotate_one_body_tensor(self.one_rdm, rotation),
            two_rdm=rotate_two_body_tensor(self.two_rdm, rotation, rotation),
        )

    def expectation(self, mol_ham: MolecularHamiltonian) -> float:
        """Return the expectation value of the RDMs with a molecular Hamiltonian."""
        return (
            mol_ham.constant
            + contract("ab,ab->", mol_ham.one_body_tensor, self.one_rdm)
            + 0.5 * contract("abcd,abcd->", mol_ham.two_body_tensor, self.two_rdm)
        ).real


def rdms(
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    rank: int = 1,
    spin_summed: bool = False,
    reorder: bool = True,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Return the reduced density matrices of a state vector.

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
        Ranks 1 through 4 are supported. A rank-``k`` spin-summed tensor stores
        ``norb ** (2*k)`` entries. For ranks 3 and 4, spin-resolved calculations
        use real intermediate tensors with ``(2*norb) ** (2*k)`` entries and a
        spinless state vector of dimension ``binom(2*norb, sum(nelec))``. These
        calculations can require substantially more memory than the returned RDMs.

        To compute matrix elements between two different states, use
        :func:`transition_rdms`.

    Args:
        vec: The state vector whose reduced density matrix is desired.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        rank: The highest rank of reduced density matrix to return, from 1 to 4.
        spin_summed: Whether to return the "spin-summed" RDMs.
        reorder: Whether to reorder the indices of the reduced density matrix.

    Returns:
        The reduced density matrices.
        All RDMs up to and including the specified rank are returned, in increasing
        order of rank. For example, if ``rank=2`` then a tuple ``(rdm1, rdm2)`` is
        returned.
        For ``rank=1``, the 1-RDM is returned directly.

        Spin-summed rank-``k`` tensors have shape ``(norb,) * (2*k)`` and sum over
        all spin assignments to the ``k`` creation-annihilation pairs. In
        particular, the spin-summed 1-RDM is alpha-alpha + beta-beta.

        Spin-resolved tensors have shape ``(k+1,) + (norb,) * (2*k)``. The leading
        axis selects the spin blocks, with ``a`` denoting alpha and ``b`` beta for
        each creation-annihilation pair:
        ``(a, b)`` for rank 1, ``(aa, ab, bb)`` for rank 2,
        ``(aaa, aab, abb, bbb)`` for rank 3, and
        ``(aaaa, aaab, aabb, abbb, bbbb)`` for rank 4.
    """
    tensors = _rdms(vec, norb, nelec, rank, spin_summed, reorder)
    return tensors[0] if rank == 1 else tensors


def transition_rdms(
    bra: np.ndarray,
    ket: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    rank: int = 1,
    spin_summed: bool = False,
    reorder: bool = True,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Return the transition reduced density matrices between two state vectors.

    Transition RDMs use the same operators and index conventions as :func:`rdms`,
    with each expectation value replaced by a matrix element between ``bra`` and
    ``ket``. For example:

    .. code::

        rdm1[p, q] = ⟨bra| p+ q |ket⟩
        rdm2[p, q, r, s] = ⟨bra| p+ r+ s q |ket⟩  (reorder=True)
        rdm2[p, q, r, s] = ⟨bra| p+ q r+ s |ket⟩  (reorder=False)

    Both inputs contain ket-state amplitudes; this function conjugates ``bra``.
    The states must have the same numbers of alpha and beta electrons. They need
    not be normalized or orthogonal, and the result is not divided by their
    overlap. Passing the same state for both arguments gives its ordinary RDMs,
    but :func:`rdms` computes those more efficiently.

    Note:
        Ranks 1 through 4 are supported, with the same memory requirements per
        real tensor family as :func:`rdms`. Spin-resolved calculations at ranks
        3 and 4 embed both states in the spinless representation.

    Args:
        bra: The state vector on the left of the matrix element.
        ket: The state vector on the right of the matrix element.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons in each state.
        rank: The highest rank of reduced density matrix to return, from 1 to 4.
        spin_summed: Whether to return the "spin-summed" RDMs.
        reorder: Whether to reorder the indices of the reduced density matrix.

    Returns:
        All transition RDMs up to and including ``rank``, in increasing order of
        rank. For ``rank=1``, the 1-RDM is returned directly; otherwise a tuple is
        returned. All tensors have complex dtype.

        Spin-summed rank-``k`` tensors have shape ``(norb,) * (2*k)``.
        Spin-resolved tensors have shape ``(k+1,) + (norb,) * (2*k)``, with the
        same spin blocks as :func:`rdms`: ``(a, b)``, ``(aa, ab, bb)``,
        ``(aaa, aab, abb, bbb)``, and ``(aaaa, aaab, aabb, abbb, bbbb)``.
    """
    if bra.shape != ket.shape:
        raise ValueError(
            "The bra and ket must have the same shape. "
            f"Got shapes {bra.shape} and {ket.shape}."
        )
    tensors = _rdms(bra, norb, nelec, rank, spin_summed, reorder, ket=ket)
    return tensors[0] if rank == 1 else tensors


def _rdms(
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    rank: int,
    spin_summed: bool,
    reorder: bool,
    *,
    ket: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    if rank not in (1, 2, 3, 4):
        raise NotImplementedError(
            f"Computing the rank {rank} reduced density matrix is currently "
            "not supported."
        )
    if rank <= 2:
        link_index = (
            gen_linkstr_index(range(norb), nelec[0]),
            gen_linkstr_index(range(norb), nelec[1]),
        )
        compute_rdms = partial(
            _rdms_low_rank_real,
            norb=norb,
            nelec=nelec,
            rank=rank,
            spin_summed=spin_summed,
            reorder=reorder,
            link_index=link_index,
            with_adjoint=ket is None,
        )
    else:
        if not spin_summed:
            vec = spinful_to_spinless_vec(vec, norb, nelec)
            if ket is not None:
                ket = spinful_to_spinless_vec(ket, norb, nelec)
            nelec = (sum(nelec), 0)
        compute_rdms = partial(
            _rdms_high_rank_real,
            norb=norb,
            nelec=nelec,
            rank=rank,
            spin_summed=spin_summed,
            reorder=reorder,
            with_adjoint=ket is None,
        )
    # PySCF's kernels read contiguous C double arrays directly.
    real = np.ascontiguousarray(vec.real, dtype=np.float64)
    imag = np.ascontiguousarray(vec.imag, dtype=np.float64)
    rdms = tuple(
        np.zeros((() if spin_summed else (k + 1,)) + (norb,) * (2 * k), dtype=complex)
        for k in range(1, rank + 1)
    )
    if ket is not None:
        ket_real = np.ascontiguousarray(ket.real, dtype=np.float64)
        ket_imag = np.ascontiguousarray(ket.imag, dtype=np.float64)
        # For bra = x + i*y and ket = u + i*v, the result is
        # T_xu + T_yv + i*T_xv - i*T_yu. Each term is listed with the signs of
        # its contributions to the real and imaginary parts.
        if real.any() and imag.any() and ket_real.any() and ket_imag.any():
            # T is bilinear, so T_{x-y,u+v} = T_xu + T_xv - T_yu - T_yv gives the
            # imaginary part as T_{x-y,u+v} - T_xu + T_yv with one fewer kernel call.
            transition_terms = [
                (real, ket_real, 1, -1),
                (imag, ket_imag, 1, 1),
                (real - imag, ket_real + ket_imag, 0, 1),
            ]
        else:
            # Skip terms with a vanishing factor.
            transition_terms = [
                (left, right, real_sign, imag_sign)
                for left, right, real_sign, imag_sign in [
                    (real, ket_real, 1, 0),
                    (imag, ket_imag, 1, 0),
                    (real, ket_imag, 0, 1),
                    (imag, ket_real, 0, -1),
                ]
                if left.any() and right.any()
            ]
        for left, right, real_sign, imag_sign in transition_terms:
            _accumulate_transition_rdms(
                rdms, compute_rdms(left, right, True), real_sign, imag_sign
            )
        return rdms

    terms = [(real, real, False)]
    # The imaginary and transition terms vanish for a real vector.
    if imag.any():
        terms += [(imag, imag, False), (real, imag, True)]
    for bra, ket, transition in terms:
        _accumulate_rdms(rdms, compute_rdms(bra, ket, transition), transition)
    return rdms


def _accumulate_rdms(
    rdms: tuple[np.ndarray, ...],
    tensors: Iterator[tuple[np.ndarray, np.ndarray]],
    transition: bool,
) -> None:
    """Accumulate one real tensor family, releasing intermediates before the next."""
    for rdm, (tensor, adjoint) in zip(rdms, tensors):
        if transition:
            # For vec = x + i*y, the imaginary contribution is i*(T_xy - T_yx).
            rdm.imag += tensor
            rdm.imag -= adjoint
        else:
            rdm.real += tensor


def _accumulate_transition_rdms(
    rdms: tuple[np.ndarray, ...],
    tensors: Iterator[tuple[np.ndarray, np.ndarray]],
    real_sign: int,
    imag_sign: int,
) -> None:
    """Accumulate a transition term, releasing intermediates before the next.

    The term is added to the real and imaginary parts with signs ``real_sign`` and
    ``imag_sign``, each of which is 1, -1, or 0 to skip that part.
    """
    for rdm, (tensor, _) in zip(rdms, tensors):
        for part, sign in [(rdm.real, real_sign), (rdm.imag, imag_sign)]:
            if sign == 1:
                part += tensor
            elif sign == -1:
                part -= tensor


def _rdms_low_rank_real(
    bra: np.ndarray,
    ket: np.ndarray,
    transition: bool,
    *,
    norb: int,
    nelec: tuple[int, int],
    rank: int,
    spin_summed: bool,
    reorder: bool,
    link_index: tuple[np.ndarray, np.ndarray],
    with_adjoint: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield real tensors and their adjoints from the direct rank-1/rank-2 kernels.

    If ``with_adjoint`` is False, the spin-resolved rank-2 transition adjoint is not
    formed and the tensor itself is yielded in its place.
    """
    if rank == 1:
        make, trans = (
            (make_rdm1, trans_rdm1) if spin_summed else (make_rdm1s, trans_rdm1s)
        )
    else:
        make, trans = (
            (make_rdm12, trans_rdm12) if spin_summed else (make_rdm12s, trans_rdm12s)
        )
        make = partial(make, reorder=reorder)
        trans = partial(trans, reorder=reorder)
    if transition:
        tensors = trans(bra, ket, norb, nelec, link_index=link_index)
    else:
        tensors = make(bra, norb, nelec, link_index=link_index)
    if rank == 1:
        tensors = (tensors,)

    if spin_summed:
        for k, tensor in enumerate(tensors, start=1):
            # PySCF returns the 1-RDM in the transposed convention.
            if k == 1:
                tensor = tensor.T
            yield tensor, tensor.T
    else:
        rdm1 = np.stack(tensors[0])
        yield rdm1.transpose(0, 2, 1), rdm1
        if rank == 2:
            if transition:
                aa, ab, _, bb = tensors[1]
            else:
                aa, ab, bb = tensors[1]
            rdm2 = np.stack([aa, ab, bb])
            adjoint = rdm2
            if transition and with_adjoint:
                # Alpha and beta pairs commute, so restore the canonical ab
                # block after reversing the operator order for the adjoint.
                adjoint = np.stack([aa.T, ab.transpose(1, 0, 3, 2), bb.T])
            yield rdm2, adjoint


def _rdms_high_rank_real(
    bra: np.ndarray,
    ket: np.ndarray,
    transition: bool,
    *,
    norb: int,
    nelec: tuple[int, int],
    rank: int,
    spin_summed: bool,
    reorder: bool,
    with_adjoint: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield real tensors and their adjoints from the spin-free higher-rank kernels.

    If ``with_adjoint`` is False, transition adjoints are not formed and each tensor
    itself is yielded in its place.
    """
    kernel_norb = norb if spin_summed else 2 * norb
    if rank == 3:
        tensors = make_dm123("FCI3pdm_kern_sf", bra, ket, kernel_norb, nelec)
        if reorder:
            tensors = reorder_dm123(*tensors)
    else:
        tensors = make_dm1234("FCI4pdm_kern_sf", bra, ket, kernel_norb, nelec)
        if reorder:
            tensors = reorder_dm1234(*tensors)

    for k, tensor in enumerate(tensors, start=1):
        # PySCF returns the 1-RDM in the transposed convention.
        if k == 1:
            tensor = tensor.T
        contribution = tensor if spin_summed else _extract_spin_blocks(tensor, norb)
        adjoint = contribution
        if transition and with_adjoint:
            # Reversing all axes takes the adjoint of the operator in either
            # ordering convention. Extract after reversal to retain the requested
            # spin blocks without forming a complex spin-orbital tensor.
            adjoint = tensor.T if spin_summed else _extract_spin_blocks(tensor.T, norb)
        yield contribution, adjoint


def _extract_spin_blocks(tensor: np.ndarray, norb: int) -> np.ndarray:
    """Extract blocks with all alpha pairs before all beta pairs."""
    rank = tensor.ndim // 2
    alpha = (slice(0, norb),) * 2
    beta = (slice(norb, 2 * norb),) * 2
    return np.stack(
        [tensor[alpha * (rank - n_beta) + beta * n_beta] for n_beta in range(rank + 1)]
    )
