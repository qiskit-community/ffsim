# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fermi-Hubbard operator."""

from ffsim._lib import FermionOperator
from ffsim.operators.fermion_action import cre_a, cre_b, des_a, des_b


def fermi_hubbard(norb: int, t: float, U: float, mu: float, V: float, PBC: bool = False) -> FermionOperator:
    r"""Fermi-Hubbard operator.

    The Fermi-Hubbard operator for :math:`N` spatial orbitals is defined as

    .. math::

        H = -t \sum_{p=0}^{N-1} \sum_{\sigma} (a^\dagger_{p+1, \sigma} a_{p, \sigma} + \text{H.c.})
        + \frac{U}{2} \sum_{p=0}^{N} \sum_{\sigma, \sigma'} a^\dagger_{p, \sigma} a^\dagger_{p, \sigma'} a_{p, \sigma'} a_{p, \sigma}
        - \mu \sum_{p=0}^N \sum_{\sigma} a^\dagger_{\sigma, p} a_{\sigma, p}
        + V \sum_{p=0}^{N-1} \sum_{\sigma, \sigma'} a^\dagger_{p, \sigma} a^\dagger_{p+1, \sigma'} a_{p+1, \sigma'} a_{p, \sigma}

    Args:
        norb: The number of spatial orbitals.
        t: The hopping strength.
        U: The onsite interaction strength.
        mu: The chemical potential.
        V: The offsite interaction strength.
        PBC: The periodic boundary conditions flag.

    Returns:
        The Fermi-Hubbard operator for the specified number of orbitals.
    """
    coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {}

    for p in range(norb-int(not PBC)):
        coeffs[(cre_a((p+1) % norb), des_a(p))] = -t
        coeffs[(cre_b((p+1) % norb), des_b(p))] = -t
        coeffs[(cre_a(p), des_a((p+1) % norb))] = -t
        coeffs[(cre_b(p), des_b((p+1) % norb))] = -t
        coeffs[(cre_a(p), cre_a((p+1) % norb), des_a((p+1) % norb), des_a(p))] = V
        coeffs[(cre_a(p), cre_b((p+1) % norb), des_b((p+1) % norb), des_a(p))] = V
        coeffs[(cre_b(p), cre_a((p+1) % norb), des_a((p+1) % norb), des_b(p))] = V
        coeffs[(cre_b(p), cre_b((p+1) % norb), des_b((p+1) % norb), des_b(p))] = V

    for p in range(norb):
        coeffs[(cre_a(p), cre_b(p), des_b(p), des_a(p))] = U/2
        coeffs[(cre_b(p), cre_a(p), des_a(p), des_b(p))] = U/2
        coeffs[(cre_a(p), des_a(p))] = -mu
        coeffs[(cre_b(p), des_b(p))] = -mu

    return FermionOperator(coeffs)
