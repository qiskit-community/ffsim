# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Time evolution by linear combination of number operators."""

from __future__ import annotations

import math
from typing import Optional, cast, overload

import numpy as np

from ffsim._lib import apply_num_op_sum_evolution_in_place
from ffsim.cistring import gen_occslst
from ffsim.gates.orbital_rotation import apply_orbital_rotation


def _conjugate_orbital_rotation(
    orbital_rotation: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
) -> np.ndarray | tuple[np.ndarray | None, np.ndarray | None]:
    if isinstance(orbital_rotation, np.ndarray) and orbital_rotation.ndim == 2:
        return orbital_rotation.T.conj()
    orbital_rotation_a, orbital_rotation_b = orbital_rotation
    if orbital_rotation_a is not None:
        orbital_rotation_a = orbital_rotation_a.T.conj()
    if orbital_rotation_b is not None:
        orbital_rotation_b = orbital_rotation_b.T.conj()
    return (orbital_rotation_a, orbital_rotation_b)


@overload
def apply_num_op_sum_evolution(
    vec: np.ndarray,
    coeffs: np.ndarray,
    time: float,
    norb: int,
    nelec: int,
    *,
    orbital_rotation: np.ndarray | None = None,
    copy: bool = True,
) -> np.ndarray: ...
@overload
def apply_num_op_sum_evolution(
    vec: np.ndarray,
    coeffs: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
    time: float,
    norb: int,
    nelec: tuple[int, int],
    *,
    orbital_rotation: np.ndarray
    | tuple[np.ndarray | None, np.ndarray | None]
    | None = None,
    copy: bool = True,
) -> np.ndarray: ...
def apply_num_op_sum_evolution(
    vec: np.ndarray,
    coeffs: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
    time: float,
    norb: int,
    nelec: int | tuple[int, int],
    *,
    orbital_rotation: np.ndarray
    | tuple[np.ndarray | None, np.ndarray | None]
    | None = None,
    copy: bool = True,
) -> np.ndarray:
    r"""Apply time evolution by a (rotated) linear combination of number operators.

    Applies

    .. math::

        \mathcal{U}
        \exp\left(-i t \sum_{\sigma, i} \lambda^{(\sigma)}_i n_{\sigma, i}\right)
        \mathcal{U}^\dagger

    where :math:`n_{\sigma, i}` denotes the number operator on orbital :math:`i`
    with spin :math:`\sigma`, the :math:`\lambda_i` are real numbers, and
    :math:`\mathcal{U}` is an optional orbital rotation.

    Args:
        vec: The state vector to be transformed.
        coeffs: The coefficients of the linear combination.
            You can pass either a single Numpy array specifying the coefficients
            to apply to both spin sectors, or you can pass a pair of Numpy arrays
            specifying independent coefficients for spin alpha and spin beta.
            If passing a pair, you can use ``None`` for one of the
            values in the pair to indicate that no operation should be applied to that
            spin sector.
        time: The evolution time.
        norb: The number of spatial orbitals.
        nelec: Either a single integer representing the number of fermions for a
            spinless system, or a pair of integers storing the numbers of spin alpha
            and spin beta fermions.
        orbital_rotation: The optional orbital rotation.
            You can pass either a single Numpy array specifying the orbital rotation
            to apply to both spin sectors, or you can pass a pair of Numpy arrays
            specifying independent orbital rotations for spin alpha and spin beta.
            If passing a pair, you can use ``None`` for one of the
            values in the pair to indicate that no operation should be applied to that
            spin sector.
        copy: Whether to copy the vector before operating on it.

            - If `copy=True` then this function always returns a newly allocated
              vector and the original vector is left untouched.
            - If `copy=False` then this function may still return a newly allocated
              vector, but the original vector may have its data overwritten.
              It is also possible that the original vector is returned,
              modified in-place.

    Returns:
        The evolved state vector.
    """
    if copy:
        vec = vec.copy()
    if isinstance(nelec, int):
        return _apply_num_op_sum_evolution_spinless(
            vec=vec,
            coeffs=cast(np.ndarray, coeffs),
            time=time,
            norb=norb,
            nelec=nelec,
            orbital_rotation=cast(Optional[np.ndarray], orbital_rotation),
        )
    return _apply_num_op_sum_evolution_spinful(
        vec=vec,
        coeffs=coeffs,
        time=time,
        norb=norb,
        nelec=nelec,
        orbital_rotation=orbital_rotation,
    )


def _apply_num_op_sum_evolution_spinless(
    vec: np.ndarray,
    coeffs: np.ndarray,
    time: float,
    norb: int,
    nelec: int,
    orbital_rotation: np.ndarray | None,
) -> np.ndarray:
    phases = np.exp(-1j * time * coeffs)
    occupations = gen_occslst(range(norb), nelec)
    if orbital_rotation is not None:
        vec = apply_orbital_rotation(
            vec,
            orbital_rotation.T.conj(),
            norb,
            nelec,
            copy=False,
        )
    vec = vec.reshape((-1, 1))
    apply_num_op_sum_evolution_in_place(vec, phases, occupations=occupations)
    vec = vec.reshape(-1)
    if orbital_rotation is not None:
        vec = apply_orbital_rotation(
            vec,
            orbital_rotation,
            norb,
            nelec,
            copy=False,
        )
    return vec


def _apply_num_op_sum_evolution_spinful(
    vec: np.ndarray,
    coeffs: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
    time: float,
    norb: int,
    nelec: tuple[int, int],
    orbital_rotation: np.ndarray | tuple[np.ndarray | None, np.ndarray | None] | None,
) -> np.ndarray:
    phases_a, phases_b = _get_phases(coeffs, time=time)
    n_alpha, n_beta = nelec
    dim_a = math.comb(norb, n_alpha)
    dim_b = math.comb(norb, n_beta)
    occupations_a = gen_occslst(range(norb), n_alpha)
    occupations_b = gen_occslst(range(norb), n_beta)

    if orbital_rotation is not None:
        vec = apply_orbital_rotation(
            vec,
            _conjugate_orbital_rotation(orbital_rotation),
            norb,
            nelec,
            copy=False,
        )

    vec = vec.reshape((dim_a, dim_b))
    if phases_a is not None:
        # apply alpha
        apply_num_op_sum_evolution_in_place(vec, phases_a, occupations=occupations_a)
    if phases_b is not None:
        # apply beta
        vec = vec.T
        apply_num_op_sum_evolution_in_place(vec, phases_b, occupations=occupations_b)
        vec = vec.T
    vec = vec.reshape(-1)

    if orbital_rotation is not None:
        vec = apply_orbital_rotation(
            vec,
            orbital_rotation,
            norb,
            nelec,
            copy=False,
        )

    return vec


def _get_phases(
    coeffs: np.ndarray | tuple[np.ndarray | None, np.ndarray | None], time: float
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if isinstance(coeffs, np.ndarray):
        phases = np.exp(-1j * time * coeffs)
        return phases, phases
    else:
        coeffs_a, coeffs_b = coeffs
        phases_a = None
        phases_b = None
        if coeffs_a is not None:
            phases_a = np.exp(-1j * time * coeffs_a)
        if coeffs_b is not None:
            phases_b = np.exp(-1j * time * coeffs_b)
        return phases_a, phases_b
