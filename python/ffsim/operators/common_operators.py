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
        coeffs[(cre_a(orb), des_a(orb))] = 1
    if spin & Spin.BETA:
        coeffs[(cre_b(orb), des_b(orb))] = 1
    return FermionOperator(coeffs)
