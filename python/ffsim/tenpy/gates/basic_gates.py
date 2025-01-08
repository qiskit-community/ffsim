# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""TeNPy basic gates."""

import cmath
import math

import numpy as np

from ffsim.spin import Spin

# ignore lowercase variable checks to maintain TeNPy naming conventions
# ruff: noqa: N806


def _sym_cons_basis(gate: np.ndarray) -> np.ndarray:
    r"""Convert a gate to the TeNPy (N, Sz)-symmetry-conserved basis.

    Args:
        gate: The quantum gate.

    Returns:
        The quantum gate in the TeNPy (N, Sz)-symmetry-conserved basis.
    """

    # convert to (N, Sz)-symmetry-conserved basis
    if gate.shape == (4, 4):  # single-site gate
        # swap = [1, 3, 0, 2]
        perm = [2, 0, 3, 1]
    elif gate.shape == (16, 16):  # two-site gate
        # swap = [5, 11, 2, 7, 12, 15, 9, 14, 1, 6, 0, 3, 8, 13, 4, 10]
        perm = [10, 8, 2, 11, 14, 0, 9, 3, 12, 6, 15, 1, 4, 13, 7, 5]
    else:
        raise ValueError(
            "only single-site and two-site gates implemented for symmetry basis "
            "conversion"
        )

    return gate[perm][:, perm]


def givens_rotation(
    theta: float, spin: Spin = Spin.ALPHA_AND_BETA, *, phi: float = 0.0
) -> np.ndarray:
    r"""The Givens rotation gate.

    The Givens rotation gate defined in :func:`~ffsim.apply_givens_rotation`,
    returned in the TeNPy (N, Sz)-symmetry-conserved basis.

    Args:
        theta: The rotation angle.
        spin: Choice of spin sector(s) to act on.

            - To act on only spin alpha, pass :const:`ffsim.Spin.ALPHA`.
            - To act on only spin beta, pass :const:`ffsim.Spin.BETA`.
            - To act on both spin alpha and spin beta, pass
              :const:`ffsim.Spin.ALPHA_AND_BETA`.
        phi: The phase angle.

    Returns:
        The Givens rotation gate in the TeNPy (N, Sz)-symmetry-conserved basis.
    """

    # define parameters
    c = math.cos(theta)
    s = -cmath.exp(-1j * phi) * math.sin(theta)

    # alpha sector / up spins
    if spin in [Spin.ALPHA, Spin.ALPHA_AND_BETA]:
        # # using TeNPy SpinHalfFermionSite(cons_N=None, cons_Sz=None) operators
        # Ggate_a = (
        #     np.kron(sp.linalg.expm(1j * phi * Nu), Id)
        #     @ sp.linalg.expm(
        #         theta * (np.kron(Cdu @ JWd, Cu @ Id) - np.kron(Cu @ JWd, Cdu @ Id))
        #     )
        #     @ np.kron(sp.linalg.expm(-1j * phi * Nu), Id)
        # )
        Ggate_a = np.diag(
            np.array([1, c, 1, c, c, 1, c, 1, 1, c, 1, c, c, 1, c, 1], dtype=complex)
        )
        Ggate_a[1, 4] = s
        Ggate_a[3, 6] = s
        Ggate_a[9, 12] = -s
        Ggate_a[11, 14] = -s
        Ggate_a[4, 1] = -s.conjugate()
        Ggate_a[6, 3] = -s.conjugate()
        Ggate_a[12, 9] = s.conjugate()
        Ggate_a[14, 11] = s.conjugate()

    # beta sector / down spins
    if spin in [Spin.BETA, Spin.ALPHA_AND_BETA]:
        # # using TeNPy SpinHalfFermionSite(cons_N=None, cons_Sz=None) operators
        # Ggate_b = (
        #     np.kron(sp.linalg.expm(1j * phi * Nd), Id)
        #     @ sp.linalg.expm(
        #         theta * (np.kron(Cdd @ JWu, Cd @ Id) - np.kron(Cd @ JWu, Cdd @ Id))
        #     )
        #     @ np.kron(sp.linalg.expm(-1j * phi * Nd), Id)
        # )
        Ggate_b = np.diag(
            np.array([1, 1, c, c, 1, 1, c, c, c, c, 1, 1, c, c, 1, 1], dtype=complex)
        )
        Ggate_b[2, 8] = s
        Ggate_b[3, 9] = -s
        Ggate_b[6, 12] = s
        Ggate_b[7, 13] = -s
        Ggate_b[8, 2] = -s.conjugate()
        Ggate_b[9, 3] = s.conjugate()
        Ggate_b[12, 6] = -s.conjugate()
        Ggate_b[13, 7] = s.conjugate()

    # define total gate
    if spin is Spin.ALPHA:
        Ggate = Ggate_a
    elif spin is Spin.BETA:
        Ggate = Ggate_b
    elif spin is Spin.ALPHA_AND_BETA:
        Ggate = Ggate_a @ Ggate_b
    else:
        raise ValueError("undefined spin")

    # convert to (N, Sz)-symmetry-conserved basis
    Ggate_sym = _sym_cons_basis(Ggate)

    return Ggate_sym


def num_interaction(theta: float, spin: Spin = Spin.ALPHA_AND_BETA) -> np.ndarray:
    r"""The number interaction gate.

    The number interaction gate defined in :func:`~ffsim.apply_num_interaction`,
    returned in the TeNPy (N, Sz)-symmetry-conserved basis.

    Args:
        theta: The rotation angle.
        spin: Choice of spin sector(s) to act on.

            - To act on only spin alpha, pass :const:`ffsim.Spin.ALPHA`.
            - To act on only spin beta, pass :const:`ffsim.Spin.BETA`.
            - To act on both spin alpha and spin beta, pass
              :const:`ffsim.Spin.ALPHA_AND_BETA` (this is the default value).

    Returns:
        The number interaction gate in the TeNPy (N, Sz)-symmetry-conserved basis.
    """

    # define parameters
    e = cmath.exp(1j * theta)

    # alpha sector / up spins
    if spin in [Spin.ALPHA, Spin.ALPHA_AND_BETA]:
        # # using TeNPy SpinHalfFermionSite(cons_N=None, cons_Sz=None) operators
        # Ngate_a = sp.linalg.expm(1j * theta * Nu)
        Ngate_a = np.diag([1, e, 1, e])

    # beta sector / down spins
    if spin in [Spin.BETA, Spin.ALPHA_AND_BETA]:
        # # using TeNPy SpinHalfFermionSite(cons_N=None, cons_Sz=None) operators
        # Ngate_b = sp.linalg.expm(1j * theta * Nd)
        Ngate_b = np.diag([1, 1, e, e])

    # define total gate
    if spin is Spin.ALPHA:
        Ngate = Ngate_a
    elif spin is Spin.BETA:
        Ngate = Ngate_b
    elif spin is Spin.ALPHA_AND_BETA:
        Ngate = Ngate_a @ Ngate_b
    else:
        raise ValueError("undefined spin")

    # convert to (N, Sz)-symmetry-conserved basis
    Ngate_sym = _sym_cons_basis(Ngate)

    return Ngate_sym


def on_site_interaction(theta: float) -> np.ndarray:
    r"""The on-site interaction gate.

    The on-site interaction gate defined in :func:`~ffsim.apply_on_site_interaction`,
    returned in the TeNPy (N, Sz)-symmetry-conserved basis.

    Args:
        theta: The rotation angle.

    Returns:
        The on-site interaction gate in the TeNPy (N, Sz)-symmetry-conserved basis.
    """

    # # using TeNPy SpinHalfFermionSite(cons_N=None, cons_Sz=None) operators
    # OSgate = sp.linalg.expm(1j * theta * Nu @ Nd)
    e = cmath.exp(1j * theta)
    OSgate = np.diag([1, 1, 1, e])

    # convert to (N, Sz)-symmetry-conserved basis
    OSgate_sym = _sym_cons_basis(OSgate)

    return OSgate_sym


def num_num_interaction(theta: float, spin: Spin = Spin.ALPHA_AND_BETA) -> np.ndarray:
    r"""The number-number interaction gate.

    The number-number interaction gate defined in
    :func:`~ffsim.apply_num_num_interaction`, returned in the TeNPy
    (N, Sz)-symmetry-conserved basis.

    Args:
        theta: The rotation angle.
        spin: Choice of spin sector(s) to act on.

            - To act on only spin alpha, pass :const:`ffsim.Spin.ALPHA`.
            - To act on only spin beta, pass :const:`ffsim.Spin.BETA`.
            - To act on both spin alpha and spin beta, pass
              :const:`ffsim.Spin.ALPHA_AND_BETA` (this is the default value).

    Returns:
        The number-number interaction gate in the TeNPy (N, Sz)-symmetry-conserved
        basis.
    """

    # define parameters
    e = cmath.exp(1j * theta)

    # alpha sector / up spins
    if spin in [Spin.ALPHA, Spin.ALPHA_AND_BETA]:
        # # using TeNPy SpinHalfFermionSite(cons_N=None, cons_Sz=None) operators
        # NNgate_a = sp.linalg.expm(1j * theta * np.kron(Nu, Nu))
        NNgate_a = np.diag([1, 1, 1, 1, 1, e, 1, e, 1, 1, 1, 1, 1, e, 1, e])

    # beta sector / down spins
    if spin in [Spin.BETA, Spin.ALPHA_AND_BETA]:
        # # using TeNPy SpinHalfFermionSite(cons_N=None, cons_Sz=None) operators
        # NNgate_b = sp.linalg.expm(1j * theta * np.kron(Nd, Nd))
        NNgate_b = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, e, e, 1, 1, e, e])

    # define total gate
    if spin is Spin.ALPHA:
        NNgate = NNgate_a
    elif spin is Spin.BETA:
        NNgate = NNgate_b
    elif spin is Spin.ALPHA_AND_BETA:
        NNgate = NNgate_a @ NNgate_b
    else:
        raise ValueError("undefined spin")

    # convert to (N, Sz)-symmetry-conserved basis
    NNgate_sym = _sym_cons_basis(NNgate)

    return NNgate_sym
