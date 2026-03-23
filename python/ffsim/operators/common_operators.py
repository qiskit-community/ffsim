# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Common fermionic operators."""

from ffsim._lib import FermionOperator
from ffsim.operators.fermion_action import cre_a, cre_b, des_a, des_b
from ffsim.spin import Spin


def s_plus_operator(norb: int) -> FermionOperator:
    r"""Spin raising operator.

    The spin raising operator is defined as

    .. math::

        S_+ = \sum_{i=0}^{N-1} a^\dagger_{\alpha, i} a_{\beta, i}

    where :math:`N` is the number of spatial orbitals.

    Args:
        norb: The number of spatial orbitals.

    Returns:
        The spin raising operator.
    """
    coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
    for i in range(norb):
        coeffs[cre_a(i), des_b(i)] = 1
    return FermionOperator(coeffs)


def s_minus_operator(norb: int) -> FermionOperator:
    r"""Spin lowering operator.

    The spin lowering operator is defined as

    .. math::

        S_- = \sum_{i=0}^{N-1} a^\dagger_{\beta, i} a_{\alpha, i}

    where :math:`N` is the number of spatial orbitals.

    Args:
        norb: The number of spatial orbitals.

    Returns:
        The spin lowering operator.
    """
    coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
    for i in range(norb):
        coeffs[cre_b(i), des_a(i)] = 1
    return FermionOperator(coeffs)


def s_x_operator(norb: int) -> FermionOperator:
    r"""Spin X operator.

    The spin X operator is defined as

    .. math::

        S_x = \frac{1}{2}(S_+ + S_-)
            = \frac{1}{2} \sum_{i=0}^{N-1}
              \left( a^\dagger_{\alpha, i} a_{\beta, i}
              + a^\dagger_{\beta, i} a_{\alpha, i} \right)

    where :math:`N` is the number of spatial orbitals.

    Args:
        norb: The number of spatial orbitals.

    Returns:
        The spin X operator.
    """
    coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
    for i in range(norb):
        coeffs[cre_a(i), des_b(i)] = 0.5
        coeffs[cre_b(i), des_a(i)] = 0.5
    return FermionOperator(coeffs)


def s_y_operator(norb: int) -> FermionOperator:
    r"""Spin Y operator.

    The spin Y operator is defined as

    .. math::

        S_y = \frac{1}{2i}(S_+ - S_-)
            = \frac{1}{2i} \sum_{i=0}^{N-1}
              \left( a^\dagger_{\alpha, i} a_{\beta, i}
              - a^\dagger_{\beta, i} a_{\alpha, i} \right)

    where :math:`N` is the number of spatial orbitals.

    Args:
        norb: The number of spatial orbitals.

    Returns:
        The spin Y operator.
    """
    coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
    for i in range(norb):
        coeffs[cre_a(i), des_b(i)] = -0.5j
        coeffs[cre_b(i), des_a(i)] = 0.5j
    return FermionOperator(coeffs)


def s_z_operator(norb: int) -> FermionOperator:
    r"""Spin Z operator.

    The spin Z operator is defined as

    .. math::

        S_z = \frac{1}{2} \sum_{i=0}^{N-1}
              \left( a^\dagger_{\alpha, i} a_{\alpha, i}
              - a^\dagger_{\beta, i} a_{\beta, i} \right)

    where :math:`N` is the number of spatial orbitals.

    Args:
        norb: The number of spatial orbitals.

    Returns:
        The spin Z operator.
    """
    coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
    for i in range(norb):
        coeffs[cre_a(i), des_a(i)] = 0.5
        coeffs[cre_b(i), des_b(i)] = -0.5
    return FermionOperator(coeffs)


def s_squared_operator(norb: int) -> FermionOperator:
    r"""Total spin operator :math:`S^2`.

    The total spin operator is defined as

    .. math::

        S^2 = S_x^2 + S_y^2 + S_z^2 = S_- S_+ + S_z (S_z + 1)

    Args:
        norb: The number of spatial orbitals.

    Returns:
        The total spin operator :math:`S^2`.
    """
    identity = FermionOperator({(): 1.0})
    sz = s_z_operator(norb)
    return s_minus_operator(norb) * s_plus_operator(norb) + sz * (sz + identity)


def number_operator(orb: int, spin: Spin = Spin.ALPHA_AND_BETA) -> FermionOperator:
    r"""Occupation number operator.

    The occupation number operator for orbital :math:`p` is defined as

    .. math::

        n_p = \sum_\sigma a^\dagger_{\sigma, p} a_{\sigma, p}

    Args:
        orb: The orbital.
        spin: Choice of spin sector(s) to act on.

            - To act on only spin alpha, pass :const:`ffsim.Spin.ALPHA`.
            - To act on only spin beta, pass :const:`ffsim.Spin.BETA`.
            - To act on both spin alpha and spin beta, pass
              :const:`ffsim.Spin.ALPHA_AND_BETA` (this is the default value).

    Returns:
        The number operator acting on the specified orbital and spin sector(s).
    """
    coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
    if spin & Spin.ALPHA:
        coeffs[cre_a(orb), des_a(orb)] = 1
    if spin & Spin.BETA:
        coeffs[cre_b(orb), des_b(orb)] = 1
    return FermionOperator(coeffs)
