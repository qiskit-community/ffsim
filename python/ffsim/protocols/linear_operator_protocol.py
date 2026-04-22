# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Linear operator protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from pyscf.fci import cistring
from scipy.sparse.linalg import LinearOperator

from ffsim._cistring import make_strings
from ffsim._lib import contract_fermion_operator_into_buffer
from ffsim.operators import FermionOperator
from ffsim.states.dimensions import dim, dims


class SupportsLinearOperator(Protocol):
    """An object that can be converted to a SciPy LinearOperator."""

    def _linear_operator_(
        self, norb: int, nelec: int | tuple[int, int]
    ) -> LinearOperator:
        """Return a SciPy LinearOperator representing the object.

        Args:
            norb: The number of spatial orbitals.
            nelec: The number of alpha and beta electrons.

        Returns:
            A Scipy LinearOperator representing the object.
        """


@dataclass(frozen=True)
class _SpinTermData:
    src_addrs: np.ndarray
    dst_addrs: np.ndarray
    phases: np.ndarray
    is_identity: bool = False


@dataclass(frozen=True)
class _FermionTermData:
    coeff: complex
    alpha: _SpinTermData
    beta: _SpinTermData


@dataclass(frozen=True)
class _ContractionData:
    scalar_coeffs: np.ndarray
    alpha_only_src: np.ndarray
    alpha_only_dst: np.ndarray
    alpha_only_scale: np.ndarray
    beta_only_src: np.ndarray
    beta_only_dst: np.ndarray
    beta_only_scale: np.ndarray
    mixed_coeffs: np.ndarray
    mixed_alpha_indptr: np.ndarray
    mixed_alpha_src: np.ndarray
    mixed_alpha_dst: np.ndarray
    mixed_alpha_phase: np.ndarray
    mixed_beta_indptr: np.ndarray
    mixed_beta_src: np.ndarray
    mixed_beta_dst: np.ndarray
    mixed_beta_phase: np.ndarray


def linear_operator(
    obj: Any, norb: int, nelec: int | tuple[int, int]
) -> LinearOperator:
    """Return a SciPy LinearOperator representing the object.

    Args:
        obj: The object to convert to a LinearOperator.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.

    Returns:
        A Scipy LinearOperator representing the object.
    """
    if isinstance(obj, FermionOperator):
        return _fermion_operator_to_linear_operator(obj, norb=norb, nelec=nelec)

    method = getattr(obj, "_linear_operator_", None)
    if method is not None:
        return method(norb=norb, nelec=nelec)

    raise TypeError(f"Object of type {type(obj)} has no _linear_operator_ method.")


def _fermion_operator_to_linear_operator(
    operator: FermionOperator, norb: int, nelec: int | tuple[int, int]
):
    """Return a SciPy LinearOperator representing a FermionOperator.

    The operator is normal ordered and packed into transition data for fast
    contraction in a fixed particle-number and spin-Z sector.

    Args:
        operator: The FermionOperator to convert to a LinearOperator.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.

    Returns:
        A Scipy LinearOperator representing the FermionOperator.
    """
    if not (operator.conserves_particle_number() and operator.conserves_spin_z()):
        raise ValueError(
            "The given FermionOperator could not be converted to a LinearOperator "
            "because it does not conserve particle number and the Z component of spin. "
            f"Conserves particle number: {operator.conserves_particle_number()} "
            f"Conserves spin Z: {operator.conserves_spin_z()}"
        )

    if isinstance(nelec, int):
        nelec = (nelec, 0)

    dim_ = dim(norb, nelec)
    dims_ = dims(norb, nelec)
    # Drop terms outside the active orbital space.
    bounded_operator = FermionOperator(
        {
            term: coeff
            for term, coeff in operator.items()
            if all(0 <= orb < norb for _, _, orb in term)
        }
    )
    fermion_term_data: list[_FermionTermData] = []
    # Split each term into alpha and beta transition data.
    for term, coeff in bounded_operator.normal_ordered().items():
        if not coeff:
            continue

        alpha_term = []
        beta_term = []
        beta_count = 0
        inversions = 0
        for action, spin, orb in term:
            if spin:
                beta_term.append((action, spin, orb))
                beta_count += 1
            else:
                alpha_term.append((action, spin, orb))
                inversions += beta_count
        if inversions & 1:
            coeff = -coeff
        fermion_term_data.append(
            _FermionTermData(
                coeff=coeff,
                alpha=_make_spin_term_data(tuple(alpha_term), norb=norb, nocc=nelec[0]),
                beta=_make_spin_term_data(tuple(beta_term), norb=norb, nocc=nelec[1]),
            )
        )
    scalar_coeff = 0j
    alpha_only_src_parts = []
    alpha_only_dst_parts = []
    alpha_only_scale_parts = []
    beta_only_src_parts = []
    beta_only_dst_parts = []
    beta_only_scale_parts = []
    mixed_terms = []

    # Group terms by the spin sectors on which they act.
    for term_data in fermion_term_data:
        alpha_is_identity = term_data.alpha.is_identity
        beta_is_identity = term_data.beta.is_identity
        alpha_len = len(term_data.alpha.src_addrs)
        beta_len = len(term_data.beta.src_addrs)

        if alpha_is_identity and beta_is_identity:
            scalar_coeff += term_data.coeff
            continue
        if not alpha_is_identity and not alpha_len:
            continue
        if not beta_is_identity and not beta_len:
            continue

        if beta_is_identity:
            alpha_only_src_parts.append(term_data.alpha.src_addrs)
            alpha_only_dst_parts.append(term_data.alpha.dst_addrs)
            alpha_only_scale_parts.append(term_data.coeff * term_data.alpha.phases)
        elif alpha_is_identity:
            beta_only_src_parts.append(term_data.beta.src_addrs)
            beta_only_dst_parts.append(term_data.beta.dst_addrs)
            beta_only_scale_parts.append(term_data.coeff * term_data.beta.phases)
        else:
            mixed_terms.append(term_data)

    alpha_only_src, alpha_only_dst, alpha_only_scale = _combine_scaled_transitions(
        alpha_only_src_parts,
        alpha_only_dst_parts,
        alpha_only_scale_parts,
    )
    beta_only_src, beta_only_dst, beta_only_scale = _combine_scaled_transitions(
        beta_only_src_parts,
        beta_only_dst_parts,
        beta_only_scale_parts,
    )

    # Pack mixed terms into flat arrays with per-term offsets.
    n_mixed_terms = len(mixed_terms)
    mixed_coeffs = np.empty(n_mixed_terms, dtype=np.complex128)
    mixed_alpha_indptr = np.zeros(n_mixed_terms + 1, dtype=np.uintp)
    mixed_beta_indptr = np.zeros(n_mixed_terms + 1, dtype=np.uintp)

    for index, term_data in enumerate(mixed_terms):
        mixed_coeffs[index] = term_data.coeff
        mixed_alpha_indptr[index + 1] = mixed_alpha_indptr[index] + len(
            term_data.alpha.src_addrs
        )
        mixed_beta_indptr[index + 1] = mixed_beta_indptr[index] + len(
            term_data.beta.src_addrs
        )

    mixed_alpha_src = np.empty(mixed_alpha_indptr[-1], dtype=np.uintp)
    mixed_alpha_dst = np.empty(mixed_alpha_indptr[-1], dtype=np.uintp)
    mixed_alpha_phase = np.empty(mixed_alpha_indptr[-1], dtype=np.int8)
    mixed_beta_src = np.empty(mixed_beta_indptr[-1], dtype=np.uintp)
    mixed_beta_dst = np.empty(mixed_beta_indptr[-1], dtype=np.uintp)
    mixed_beta_phase = np.empty(mixed_beta_indptr[-1], dtype=np.int8)

    for index, term_data in enumerate(mixed_terms):
        alpha_start = mixed_alpha_indptr[index]
        alpha_end = mixed_alpha_indptr[index + 1]
        beta_start = mixed_beta_indptr[index]
        beta_end = mixed_beta_indptr[index + 1]

        mixed_alpha_src[alpha_start:alpha_end] = term_data.alpha.src_addrs
        mixed_alpha_dst[alpha_start:alpha_end] = term_data.alpha.dst_addrs
        mixed_alpha_phase[alpha_start:alpha_end] = term_data.alpha.phases
        mixed_beta_src[beta_start:beta_end] = term_data.beta.src_addrs
        mixed_beta_dst[beta_start:beta_end] = term_data.beta.dst_addrs
        mixed_beta_phase[beta_start:beta_end] = term_data.beta.phases

    contraction_data = _ContractionData(
        scalar_coeffs=(
            np.array([scalar_coeff], dtype=np.complex128)
            if scalar_coeff
            else np.empty(0, dtype=np.complex128)
        ),
        alpha_only_src=alpha_only_src,
        alpha_only_dst=alpha_only_dst,
        alpha_only_scale=alpha_only_scale,
        beta_only_src=beta_only_src,
        beta_only_dst=beta_only_dst,
        beta_only_scale=beta_only_scale,
        mixed_coeffs=mixed_coeffs,
        mixed_alpha_indptr=mixed_alpha_indptr,
        mixed_alpha_src=mixed_alpha_src,
        mixed_alpha_dst=mixed_alpha_dst,
        mixed_alpha_phase=mixed_alpha_phase,
        mixed_beta_indptr=mixed_beta_indptr,
        mixed_beta_src=mixed_beta_src,
        mixed_beta_dst=mixed_beta_dst,
        mixed_beta_phase=mixed_beta_phase,
    )

    def matvec(vec: np.ndarray):
        return _apply_fermion_terms(
            vec, contraction_data=contraction_data, dims_=dims_, reverse=False
        )

    def rmatvec(vec: np.ndarray):
        return _apply_fermion_terms(
            vec, contraction_data=contraction_data, dims_=dims_, reverse=True
        )

    return LinearOperator(
        shape=(dim_, dim_), matvec=matvec, rmatvec=rmatvec, dtype=complex
    )


def _make_spin_term_data(
    term: tuple[tuple[bool, bool, int], ...], norb: int, nocc: int
) -> _SpinTermData:
    """Return transition data for a spin block of a FermionOperator term.

    The empty term is treated as the identity.

    Args:
        term: The spin block of the FermionOperator term.
        norb: The number of spatial orbitals.
        nocc: The number of occupied orbitals in the spin sector.

    Returns:
        The source addresses, destination addresses, and fermionic phases of the
        spin block.
    """
    if not term:
        empty = np.array([], dtype=np.intp)
        return _SpinTermData(
            src_addrs=empty.astype(np.uintp, copy=False),
            dst_addrs=empty.astype(np.uintp, copy=False),
            phases=np.array([], dtype=np.int8),
            is_identity=True,
        )

    creation_mask = 0
    annihilation_mask = 0
    for action, _, orb in term:
        if action:
            creation_mask |= 1 << orb
        else:
            annihilation_mask |= 1 << orb
    creation_only_mask = creation_mask & ~annihilation_mask

    src_addrs = []
    dst_strings = []
    phases = []
    # prefilter source strings
    for src_addr, string in enumerate(make_strings(range(norb), nocc)):
        string = int(string)
        if string & annihilation_mask != annihilation_mask:
            continue
        if string & creation_only_mask:
            continue

        phase = 1
        transformed = string
        for action, _, orb in reversed(term):
            mask = 1 << orb
            # accumulate the fermionic phase
            if (transformed & (mask - 1)).bit_count() & 1:
                phase = -phase
            if action:
                if transformed & mask:
                    break
                transformed |= mask
            else:
                if not transformed & mask:
                    break
                transformed ^= mask
        else:
            src_addrs.append(src_addr)
            dst_strings.append(transformed)
            phases.append(phase)

    if not src_addrs:
        empty = np.array([], dtype=np.uintp)
        return _SpinTermData(
            src_addrs=empty,
            dst_addrs=empty,
            phases=np.array([], dtype=np.int8),
        )

    dst_addrs = cistring.strs2addr(
        norb=norb, nelec=nocc, strings=np.asarray(dst_strings, dtype=np.int64)
    ).astype(np.uintp, copy=False)
    return _SpinTermData(
        src_addrs=np.asarray(src_addrs, dtype=np.uintp),
        dst_addrs=dst_addrs,
        phases=np.asarray(phases, dtype=np.int8),
    )


def _combine_scaled_transitions(
    src_parts: list[np.ndarray],
    dst_parts: list[np.ndarray],
    scale_parts: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine scaled transition data into canonical flat arrays.

    Each entry represents a contribution from a source address to a destination
    address, scaled by the corresponding coefficient. Repeated source-destination
    pairs are summed, and pairs with zero total scale are removed.

    Args:
        src_parts: The source address arrays to combine.
        dst_parts: The destination address arrays to combine.
        scale_parts: The scale arrays to combine.

    Returns:
        The combined source addresses, destination addresses, and scales.
    """
    if not src_parts:
        return (
            np.empty(0, dtype=np.uintp),
            np.empty(0, dtype=np.uintp),
            np.empty(0, dtype=np.complex128),
        )

    src = np.concatenate(src_parts).astype(np.uintp, copy=False)
    dst = np.concatenate(dst_parts).astype(np.uintp, copy=False)
    scale = np.concatenate(scale_parts).astype(np.complex128, copy=False)

    order = np.lexsort((dst, src))
    src = src[order]
    dst = dst[order]
    scale = scale[order]

    change = np.empty(len(src), dtype=bool)
    change[0] = True
    change[1:] = (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])
    starts = np.flatnonzero(change)
    src = src[starts]
    dst = dst[starts]
    scale = np.add.reduceat(scale, starts)

    nonzero = scale != 0
    src = src[nonzero]
    dst = dst[nonzero]
    scale = scale[nonzero]

    order = np.lexsort((src, dst))
    return src[order], dst[order], scale[order]


def _apply_fermion_terms(
    vec: np.ndarray,
    contraction_data: _ContractionData,
    dims_: tuple[int, int],
    *,
    reverse: bool,
) -> np.ndarray:
    transformed = np.ascontiguousarray(vec, dtype=complex).reshape(dims_)
    result = np.zeros(dims_, dtype=complex)
    contract_fermion_operator_into_buffer(
        transformed,
        contraction_data.scalar_coeffs,
        contraction_data.alpha_only_src,
        contraction_data.alpha_only_dst,
        contraction_data.alpha_only_scale,
        contraction_data.beta_only_src,
        contraction_data.beta_only_dst,
        contraction_data.beta_only_scale,
        contraction_data.mixed_coeffs,
        contraction_data.mixed_alpha_indptr,
        contraction_data.mixed_alpha_src,
        contraction_data.mixed_alpha_dst,
        contraction_data.mixed_alpha_phase,
        contraction_data.mixed_beta_indptr,
        contraction_data.mixed_beta_src,
        contraction_data.mixed_beta_dst,
        contraction_data.mixed_beta_phase,
        result,
        reverse=reverse,
    )
    return result.reshape(-1)
