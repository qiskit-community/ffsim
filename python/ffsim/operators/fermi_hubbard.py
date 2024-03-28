# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fermi-Hubbard model Hamiltonian."""

from ffsim._lib import FermionOperator
from ffsim.operators.fermion_action import cre_a, cre_b, des_a, des_b


def fermi_hubbard_1d(
    norb: int,
    tunneling: float,
    interaction: float,
    chemical_potential: float = 0,
    nearest_neighbor_interaction: float = 0,
    periodic: bool = False,
) -> FermionOperator:
    r"""1D Fermi-Hubbard model Hamiltonian.

    The Hamiltonian for the 1D Fermi-Hubbard model with :math:`N` spatial orbitals is
    given by

    .. math::

        H = -t \sum_{p=1}^{N-1} \sum_{\sigma}
        (a^\dagger_{p+1, \sigma} a_{p, \sigma} + \text{H.c.})
        + \frac{U}{2} \sum_{p=1}^{N} \sum_{\sigma, \sigma'}
        a^\dagger_{p, \sigma} a^\dagger_{p, \sigma'} a_{p, \sigma'} a_{p, \sigma}
        - \mu \sum_{p=1}^N \sum_{\sigma} a^\dagger_{\sigma, p} a_{\sigma, p}
        + V \sum_{p=1}^{N-1} \sum_{\sigma, \sigma'}
        a^\dagger_{p, \sigma} a^\dagger_{p+1, \sigma'} a_{p+1, \sigma'} a_{p, \sigma}

    Args:
        norb: The number of spatial orbitals :math:`N`.
        tunneling: The tunneling amplitude :math:`t`.
        interaction: The onsite interaction strength :math:`U`.
        chemical_potential: The chemical potential :math:`\mu`.
        nearest_neighbor_interaction: The nearest-neighbor interaction strength
            :math:`V`.
        periodic: Whether to use periodic boundary conditions.

    Returns:
        The 1D Fermi-Hubbard model Hamiltonian.
    """
    coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {}

    for p in range(norb - 1 + periodic):
        coeffs[(cre_a((p + 1) % norb), des_a(p))] = -tunneling
        coeffs[(cre_b((p + 1) % norb), des_b(p))] = -tunneling
        coeffs[(cre_a(p), des_a((p + 1) % norb))] = -tunneling
        coeffs[(cre_b(p), des_b((p + 1) % norb))] = -tunneling
        if nearest_neighbor_interaction:
            coeffs[
                (cre_a(p), cre_a((p + 1) % norb), des_a((p + 1) % norb), des_a(p))
            ] = nearest_neighbor_interaction
            coeffs[
                (cre_a(p), cre_b((p + 1) % norb), des_b((p + 1) % norb), des_a(p))
            ] = nearest_neighbor_interaction
            coeffs[
                (cre_b(p), cre_a((p + 1) % norb), des_a((p + 1) % norb), des_b(p))
            ] = nearest_neighbor_interaction
            coeffs[
                (cre_b(p), cre_b((p + 1) % norb), des_b((p + 1) % norb), des_b(p))
            ] = nearest_neighbor_interaction

    for p in range(norb):
        if interaction:
            coeffs[(cre_a(p), cre_b(p), des_b(p), des_a(p))] = interaction / 2
            coeffs[(cre_b(p), cre_a(p), des_a(p), des_b(p))] = interaction / 2
        if chemical_potential:
            coeffs[(cre_a(p), des_a(p))] = -chemical_potential
            coeffs[(cre_b(p), des_b(p))] = -chemical_potential

    return FermionOperator(coeffs)
