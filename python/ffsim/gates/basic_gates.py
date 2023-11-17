# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Basic fermionic quantum computation gates."""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence

import numpy as np
from scipy.special import comb

from ffsim.gates.orbital_rotation import _one_subspace_indices, apply_orbital_rotation


def _apply_phase_shift(
    vec: np.ndarray,
    phase: complex,
    target_orbs: tuple[tuple[int, ...], tuple[int, ...]],
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
) -> np.ndarray:
    """Apply a phase shift controlled on target orbitals.

    Multiplies by the phase each coefficient corresponding to a string in which
    the target orbitals are all 1 (occupied).
    """
    if copy:
        vec = vec.copy()
    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)
    vec = vec.reshape((dim_a, dim_b))
    target_orbs_a, target_orbs_b = target_orbs
    indices_a = _one_subspace_indices(norb, n_alpha, target_orbs_a)
    indices_b = _one_subspace_indices(norb, n_beta, target_orbs_b)
    vec[np.ix_(indices_a, indices_b)] *= phase
    return vec.reshape(-1)


def apply_givens_rotation(
    vec: np.ndarray,
    theta: float,
    target_orbs: tuple[int, int],
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
):
    r"""Apply a Givens rotation gate.

    The Givens rotation gate is

    .. math::

        \text{G}(\theta, (p, q)) = \prod_{\sigma}
        \exp\left(\theta (a^\dagger_{\sigma, p} a_{\sigma, q}
        - a^\dagger_{\sigma, q} a_{\sigma, p})\right)

    Under the Jordan-Wigner transform, this gate has the following matrix when applied
    to neighboring qubits:

    .. math::

        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & \cos(\theta) & -\sin(\theta) & 0\\
            0 & \sin(\theta) & \cos(\theta) & 0\\
            0 & 0 & 0 & 1 \\
        \end{pmatrix}

    Args:
        vec: The state vector to be transformed.
        theta: The rotation angle.
        target_orbs: The orbitals (p, q) to rotate.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
            vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
            vector, but the original vector may have its data overwritten.
            It is also possible that the original vector is returned,
            modified in-place.
    """
    if len(set(target_orbs)) == 1:
        raise ValueError(f"The orbitals to rotate must be distinct. Got {target_orbs}.")
    c = math.cos(theta)
    s = math.sin(theta)
    mat = np.eye(norb)
    mat[np.ix_(target_orbs, target_orbs)] = [[c, s], [-s, c]]
    return apply_orbital_rotation(vec, mat, norb=norb, nelec=nelec, copy=copy)


def apply_tunneling_interaction(
    vec: np.ndarray,
    theta: float,
    target_orbs: tuple[int, int],
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
):
    r"""Apply a tunneling interaction gate.

    The tunneling interaction gate is

    .. math::

        \text{T}(\theta, (p, q)) = \prod_\sigma
        \exp\left(i \theta (a^\dagger_{\sigma, p} a_{\sigma, q}
        + a^\dagger_{\sigma, q} a_{\sigma, p})\right)

    Under the Jordan-Wigner transform, this gate has the following matrix when applied
    to neighboring qubits:

    .. math::

        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & \cos(\theta) & i \sin(\theta) & 0\\
            0 & i \sin(\theta) & \cos(\theta) & 0\\
            0 & 0 & 0 & 1 \\
        \end{pmatrix}

    Args:
        vec: The state vector to be transformed.
        theta: The rotation angle.
        target_orbs: The orbitals (p, q) on which to apply the interaction.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
            vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
            vector, but the original vector may have its data overwritten.
            It is also possible that the original vector is returned,
            modified in-place.
    """
    if len(set(target_orbs)) == 1:
        raise ValueError(f"The orbitals to rotate must be distinct. Got {target_orbs}.")
    vec = apply_num_interaction(
        vec, -math.pi / 2, target_orbs[0], norb=norb, nelec=nelec, copy=copy
    )
    vec = apply_givens_rotation(
        vec, theta, target_orbs, norb=norb, nelec=nelec, copy=False
    )
    vec = apply_num_interaction(
        vec, math.pi / 2, target_orbs[0], norb=norb, nelec=nelec, copy=False
    )
    return vec


def apply_num_interaction(
    vec: np.ndarray,
    theta: float,
    target_orb: int,
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
):
    r"""Apply a number interaction gate.

    The number interaction gate is

    .. math::

        \text{N}(\theta, p) = \prod_{\sigma}
        \exp\left(i \theta a^\dagger_{\sigma, p} a_{\sigma, p}\right)

    Args:
        vec: The state vector to be transformed.
        theta: The rotation angle.
        target_orb: The orbital on which to apply the interaction.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
            vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
            vector, but the original vector may have its data overwritten.
            It is also possible that the original vector is returned,
            modified in-place.
    """
    if copy:
        vec = vec.copy()
    vec = apply_num_op_prod_interaction(
        vec,
        theta,
        target_orbs=([target_orb], []),
        norb=norb,
        nelec=nelec,
        copy=False,
    )
    vec = apply_num_op_prod_interaction(
        vec,
        theta,
        target_orbs=([], [target_orb]),
        norb=norb,
        nelec=nelec,
        copy=False,
    )
    return vec


def apply_num_num_interaction(
    vec: np.ndarray,
    theta: float,
    target_orbs: tuple[int, int],
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
):
    r"""Apply a number-number interaction gate.

    The number-number interaction gate is

    .. math::

        \text{NN}(\theta, (p, q)) = \prod_{\sigma}
        \exp\left(i \theta a^\dagger_{\sigma, p} a_{\sigma, p}
        a^\dagger_{\sigma, q} a_{\sigma, q}\right)

    Args:
        vec: The state vector to be transformed.
        theta: The rotation angle.
        target_orbs: The orbitals (p, q) to interact.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
            vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
            vector, but the original vector may have its data overwritten.
            It is also possible that the original vector is returned,
            modified in-place.
    """
    if len(set(target_orbs)) == 1:
        raise ValueError(
            f"The orbitals to interact must be distinct. Got {target_orbs}."
        )
    if copy:
        vec = vec.copy()
    vec = apply_num_op_prod_interaction(
        vec,
        theta,
        target_orbs=(target_orbs, []),
        norb=norb,
        nelec=nelec,
        copy=False,
    )
    vec = apply_num_op_prod_interaction(
        vec,
        theta,
        target_orbs=([], target_orbs),
        norb=norb,
        nelec=nelec,
        copy=False,
    )
    return vec


def apply_num_op_prod_interaction(
    vec: np.ndarray,
    theta: float,
    target_orbs: tuple[Sequence[int], Sequence[int]],
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
):
    r"""Apply interaction gate for product of number operators.

    The gate is

    .. math::

        \text{NP}(\theta, (S_\alpha, S_\beta)) =
        \exp\left(i \theta
        \prod_{p \in S_\alpha} a^\dagger_{\alpha, p} a_{\alpha, p}
        \prod_{p \in S_\beta} a^\dagger_{\beta, p} a_{\beta, p}
        \right)

    Args:
        vec: The state vector to be transformed.
        theta: The rotation angle.
        target_orbs: A pair of lists of integers giving the orbitals on which to apply
            the interaction. The first list specifies the alpha orbitals and the second
            list specifies the beta orbitals.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
            vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
            vector, but the original vector may have its data overwritten.
            It is also possible that the original vector is returned,
            modified in-place.
    """
    if copy:
        vec = vec.copy()
    alpha_orbs, beta_orbs = target_orbs
    vec = _apply_phase_shift(
        vec,
        cmath.exp(1j * theta),
        (tuple(alpha_orbs), tuple(beta_orbs)),
        norb=norb,
        nelec=nelec,
        copy=False,
    )
    return vec


def apply_hop_gate(
    vec: np.ndarray,
    theta: float,
    target_orbs: tuple[int, int],
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
) -> np.ndarray:
    r"""Apply a hop gate.

    A "hop gate" is a Givens rotation gate followed by a number-number interaction with
    angle pi:

    .. math::

        \begin{align}
            \text{Hop}&(\theta, (p, q))
            = \text{NN}(\pi, (p, q)) \text{G}(\theta, (p, q)) \\
            &= \prod_{\sigma}
            \exp\left(i \theta a^\dagger_{\sigma, p} a_{\sigma, p}
            a^\dagger_{\sigma, q} a_{\sigma, q}\right)
            \exp\left(\theta (a^\dagger_{\sigma, p} a_{\sigma, q}
            - a^\dagger_{\sigma, q} a_{\sigma, p})\right)
        \end{align}

    Under the Jordan-Wigner transform, this gate has the following matrix when applied
    to neighboring qubits:

    .. math::

        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & \cos(\theta) & -\sin(\theta) & 0\\
            0 & \sin(\theta) & \cos(\theta) & 0\\
            0 & 0 & 0 & -1 \\
        \end{pmatrix}

    Args:
        vec: The state vector to be transformed.
        theta: The rotation angle.
        target_orbs: The orbitals (p, q) to interact.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
            vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
            vector, but the original vector may have its data overwritten.
            It is also possible that the original vector is returned,
            modified in-place.
    """
    if copy:
        vec = vec.copy()
    vec = apply_givens_rotation(
        vec, theta, target_orbs, norb=norb, nelec=nelec, copy=False
    )
    vec = apply_num_num_interaction(
        vec, math.pi, target_orbs, norb=norb, nelec=nelec, copy=False
    )
    return vec


def apply_fsim_gate(
    vec: np.ndarray,
    theta: float,
    phi: float,
    target_orbs: tuple[int, int],
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
) -> np.ndarray:
    r"""Apply an fSim gate.

    An fSim gate consists of a tunneling interaction followed by a number-number
    interaction (note the negative sign convention for the angles):

    .. math::

        \begin{align}
            \text{fSim}&(\theta, \phi, (p, q))
            = \text{NN}(-\phi, (p, q)) \text{T}(-\theta, (p, q)) \\
            &= \prod_\sigma
            \exp\left(-i \phi a^\dagger_{\sigma, p} a_{\sigma, p}
            a^\dagger_{\sigma, q} a_{\sigma, q}\right)
            \exp\left(-i \theta (a^\dagger_{\sigma, p} a_{\sigma, q}
            + a^\dagger_{\sigma, q} a_{\sigma, p})\right)
        \end{align}

    Under the Jordan-Wigner transform, this gate has the following matrix when applied
    to neighboring qubits:

    .. math::

        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & \cos(\theta) & -i \sin(\theta) & 0\\
            0 & -i \sin(\theta) & \cos(\theta) & 0\\
            0 & 0 & 0 & e^{-i \phi} \\
        \end{pmatrix}

    Args:
        vec: The state vector to be transformed.
        theta: The rotation angle for the tunneling interaction.
        phi: The phase angle for the number-number interaction.
        target_orbs: The orbitals (p, q) to interact.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
            vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
            vector, but the original vector may have its data overwritten.
            It is also possible that the original vector is returned,
            modified in-place.
    """
    if copy:
        vec = vec.copy()
    vec = apply_tunneling_interaction(
        vec, -theta, target_orbs, norb=norb, nelec=nelec, copy=False
    )
    vec = apply_num_num_interaction(
        vec, -phi, target_orbs, norb=norb, nelec=nelec, copy=False
    )
    return vec
