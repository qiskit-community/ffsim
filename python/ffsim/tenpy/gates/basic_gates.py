# (C) Copyright IBM 2025.
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
import scipy.linalg
from numpy.typing import NDArray

from ffsim.spin import Spin


def givens_rotation(
    theta: float, spin: Spin = Spin.ALPHA_AND_BETA, *, phi: float = 0.0
) -> NDArray[np.complex128]:
    r"""The Givens rotation gate.

    The Givens rotation gate defined in :func:`~ffsim.apply_givens_rotation`,
    returned in the TeNPy (N, Sz)-symmetry-conserved basis.

    The bitstring ordering of the TeNPy (N, Sz)-symmetry-conserved basis is:

    .. code::

        1010  # (2, -2)
        1000  # (1, -1)
        0010  # (1, -1)
        1011  # (3, -1)
        1110  # (3, -1)
        0000  # (0, 0)
        1001  # (2, 0)
        0011  # (2, 0)
        1100  # (2, 0)
        0110  # (2, 0)
        1111  # (4, 0)
        0001  # (1, 1)
        0100  # (1, 1)
        1101  # (3, 1)
        0111  # (3, 1)
        0101  # (2, 2)

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
    c = math.cos(theta)
    s = cmath.rect(1, phi) * math.sin(theta)
    mat = np.array([[c, s], [-s.conjugate(), c]])
    mat_a = mat if spin & Spin.ALPHA else np.eye(2)
    mat_b = mat if spin & Spin.BETA else np.eye(2)
    return scipy.linalg.block_diag(
        1,
        mat_b,
        mat_a.conj(),
        1,
        scipy.linalg.block_diag(mat_a.conj(), mat_a.T)[[0, 2, 1, 3]][:, [0, 2, 1, 3]]
        @ scipy.linalg.block_diag(mat_b.T.conj(), mat_b),
        1,
        mat_a.T,
        mat_b.T.conj(),
        1,
    )


def num_interaction(
    theta: float, spin: Spin = Spin.ALPHA_AND_BETA
) -> NDArray[np.complex128]:
    r"""The number interaction gate.

    The number interaction gate defined in :func:`~ffsim.apply_num_interaction`,
    returned in the TeNPy (N, Sz)-symmetry-conserved basis.

    The bitstring ordering of the TeNPy (N, Sz)-symmetry-conserved basis is:

    .. code::

        10  # (1, -1)
        00  # (0, 0)
        11  # (2, 0)
        01  # (1, 1)

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
    phase = cmath.rect(1, theta)
    alpha_phase = phase if spin & Spin.ALPHA else 1
    beta_phase = phase if spin & Spin.BETA else 1
    return np.diag([beta_phase, 1, alpha_phase * beta_phase, alpha_phase])


def on_site_interaction(theta: float) -> NDArray[np.complex128]:
    r"""The on-site interaction gate.

    The on-site interaction gate defined in :func:`~ffsim.apply_on_site_interaction`,
    returned in the TeNPy (N, Sz)-symmetry-conserved basis.

    The bitstring ordering of the TeNPy (N, Sz)-symmetry-conserved basis is:

    .. code::

        10  # (1, -1)
        00  # (0, 0)
        11  # (2, 0)
        01  # (1, 1)

    Args:
        theta: The rotation angle.

    Returns:
        The on-site interaction gate in the TeNPy (N, Sz)-symmetry-conserved basis.
    """
    return np.diag([1, 1, cmath.rect(1, theta), 1])


def num_num_interaction(
    theta: float, spin: Spin = Spin.ALPHA_AND_BETA
) -> NDArray[np.complex128]:
    r"""The number-number interaction gate.

    The number-number interaction gate defined in
    :func:`~ffsim.apply_num_num_interaction`, returned in the TeNPy
    (N, Sz)-symmetry-conserved basis.

    The bitstring ordering of the TeNPy (N, Sz)-symmetry-conserved basis is:

    .. code::

        1010  # (2, -2)
        1000  # (1, -1)
        0010  # (1, -1)
        1011  # (3, -1)
        1110  # (3, -1)
        0000  # (0, 0)
        1001  # (2, 0)
        0011  # (2, 0)
        1100  # (2, 0)
        0110  # (2, 0)
        1111  # (4, 0)
        0001  # (1, 1)
        0100  # (1, 1)
        1101  # (3, 1)
        0111  # (3, 1)
        0101  # (2, 2)

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
    phase = cmath.rect(1, theta)
    alpha_phase = phase if spin & Spin.ALPHA else 1
    beta_phase = phase if spin & Spin.BETA else 1
    return np.diag(
        [
            beta_phase,
            1,
            1,
            beta_phase,
            beta_phase,
            1,
            1,
            1,
            1,
            1,
            alpha_phase * beta_phase,
            1,
            1,
            alpha_phase,
            alpha_phase,
            alpha_phase,
        ]
    )
