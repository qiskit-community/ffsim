# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Time evolution by a quadratic Hamiltonian."""

from __future__ import annotations

from typing import cast, overload

import numpy as np
import scipy.linalg

from ffsim.gates.orbital_rotation import apply_orbital_rotation


@overload
def apply_quad_ham_evolution(
    vec: np.ndarray,
    mat: np.ndarray,
    time: float,
    norb: int,
    nelec: int,
    *,
    copy: bool = True,
) -> np.ndarray: ...
@overload
def apply_quad_ham_evolution(
    vec: np.ndarray,
    mat: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
    time: float,
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
) -> np.ndarray: ...
def apply_quad_ham_evolution(
    vec: np.ndarray,
    mat: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
    time: float,
    norb: int,
    nelec: int | tuple[int, int],
    *,
    copy: bool = True,
) -> np.ndarray:
    r"""Apply time evolution by a quadratic Hamiltonian.

    Applies

    .. math::

        \exp\left(-i t \sum_{\sigma, ij}
        \mathbf{M}^{(\sigma)}_{ij} a^\dagger_{\sigma, i} a_{\sigma, j}\right)

    where each :math:`\mathbf{M}^{(\sigma)}` is a Hermitian matrix.

    Args:
        vec: The state vector to be transformed.
        mat: The matrix :math:`\mathbf{M}` describing the quadratic Hamiltonian.
            You can pass either a single Numpy array specifying the Hamiltonian
            to use for both spin sectors, or you can pass a pair of Numpy arrays
            specifying independent Hamiltonians for spin alpha and spin beta.
            If passing a pair, you can use ``None`` for one of the values in the pair
            to indicate that no operation should be applied to that spin sector.
        time: The evolution time.
        norb: The number of spatial orbitals.
        nelec: Either a single integer representing the number of fermions for a
            spinless system, or a pair of integers storing the numbers of spin alpha
            and spin beta fermions.
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

    if isinstance(nelec, int) or (isinstance(mat, np.ndarray) and mat.ndim == 2):
        # Spinless, or spinful with same matrix for each spin sector
        evolution_mat = scipy.linalg.expm(-1j * time * cast(np.ndarray, mat))
        return apply_orbital_rotation(
            vec, evolution_mat, norb=norb, nelec=nelec, copy=False
        )

    # Spinful with different matrices for each spin sector
    mat_a, mat_b = mat
    evolution_mat_a = None
    evolution_mat_b = None
    if mat_a is not None:
        evolution_mat_a = scipy.linalg.expm(-1j * time * mat_a)
    if mat_b is not None:
        evolution_mat_b = scipy.linalg.expm(-1j * time * mat_b)
    return apply_orbital_rotation(
        vec, (evolution_mat_a, evolution_mat_b), norb=norb, nelec=nelec, copy=False
    )
