# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fermi-Hubbard model Hamiltonian."""

from collections import defaultdict

from ffsim._lib import FermionOperator
from ffsim.operators.fermion_action import cre_a, cre_b, des_a, des_b


def fermi_hubbard_1d(
    norb: int,
    tunneling: float,
    interaction: float,
    *,
    chemical_potential: float = 0,
    nearest_neighbor_interaction: float = 0,
    periodic: bool = False,
) -> FermionOperator:
    r"""One-dimensional Fermi-Hubbard model Hamiltonian.

    The Hamiltonian for the one-dimensional Fermi-Hubbard model with :math:`N` spatial
    orbitals is given by

    .. math::

        H = -t \sum_{\sigma} \sum_{p}
        (a^\dagger_{\sigma, p} a_{\sigma, p+1} + \text{h.c.})
        + U \sum_{p=1}^{N} n_{\alpha, p} n_{\beta, p}
        - \mu \sum_{p=1}^N (n_{\alpha, p} + n_{\beta, p})
        + V \sum_{\sigma, \sigma'} \sum_{p} n_{\sigma, p} n_{\sigma', p+1}

    where :math:`n_{\sigma, p} = a_{\sigma, p}^\dagger a_{\sigma, p}` is the number
    operator on orbital :math:`p` with spin :math:`\sigma`.

    For the tunneling and nearest-neighbor interaction terms, the summations over
    :math:`p` run along edges in the network. Under open boundary conditions, there are
    :math:`N` vertices and :math:`N-1` edges connected in a chain and hence, the
    summations over :math:`p` are defined as :math:`\sum_{p}=\sum_{p=1}^{N-1}`. Under
    periodic boundary conditions, there are :math:`N` vertices and :math:`N` edges
    connected in a ring and hence, the summations over :math:`p` are defined as
    :math:`\sum_{p}=\sum_{p=1}^{N}`, with operator positions :math:`p+1\to(p+1)\bmod N`.

    References:
        - `The Hubbard Model`_

    Args:
        norb: The number of spatial orbitals :math:`N`.
        tunneling: The tunneling amplitude :math:`t`.
        interaction: The onsite interaction strength :math:`U`.
        chemical_potential: The chemical potential :math:`\mu`.
        nearest_neighbor_interaction: The nearest-neighbor interaction strength
            :math:`V`.
        periodic: Whether to use periodic boundary conditions.

    Returns:
        The one-dimensional Fermi-Hubbard model Hamiltonian.

    .. _The Hubbard Model: https://doi.org/10.1146/annurev-conmatphys-031620-102024
    """
    coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = defaultdict(float)

    for orb in range(norb):
        if tunneling and (orb != norb - 1 or periodic):
            coeffs[cre_a(orb), des_a((orb + 1) % norb)] -= tunneling
            coeffs[cre_b(orb), des_b((orb + 1) % norb)] -= tunneling
            coeffs[cre_a((orb + 1) % norb), des_a(orb)] -= tunneling
            coeffs[cre_b((orb + 1) % norb), des_b(orb)] -= tunneling
        if nearest_neighbor_interaction and (orb != norb - 1 or periodic):
            coeffs[
                cre_a(orb), des_a(orb), cre_a((orb + 1) % norb), des_a((orb + 1) % norb)
            ] += nearest_neighbor_interaction
            coeffs[
                cre_a(orb), des_a(orb), cre_b((orb + 1) % norb), des_b((orb + 1) % norb)
            ] += nearest_neighbor_interaction
            coeffs[
                cre_b(orb), des_b(orb), cre_a((orb + 1) % norb), des_a((orb + 1) % norb)
            ] += nearest_neighbor_interaction
            coeffs[
                cre_b(orb), des_b(orb), cre_b((orb + 1) % norb), des_b((orb + 1) % norb)
            ] += nearest_neighbor_interaction
        if interaction:
            coeffs[cre_a(orb), des_a(orb), cre_b(orb), des_b(orb)] = interaction
        if chemical_potential:
            coeffs[cre_a(orb), des_a(orb)] = -chemical_potential
            coeffs[cre_b(orb), des_b(orb)] = -chemical_potential

    return FermionOperator(coeffs)


def fermi_hubbard_2d(
    norb_x: int,
    norb_y: int,
    tunneling: float,
    interaction: float,
    *,
    chemical_potential: float = 0,
    nearest_neighbor_interaction: float = 0,
    periodic: bool = False,
) -> FermionOperator:
    r"""Two-dimensional Fermi-Hubbard model Hamiltonian.

    The Hamiltonian for the two-dimensional Fermi-Hubbard model on a square lattice with
    :math:`N = N_x \times N_y` spatial orbitals is given by

    .. math::

        H = -t \sum_{\sigma} \sum_{\braket{pq}}
        (a^\dagger_{\sigma, p} a_{\sigma, q} + \text{h.c.})
        + U \sum_{p=1}^{N} n_{\alpha, p} n_{\beta, p}
        - \mu \sum_{p=1}^{N} (n_{\alpha, p} + n_{\beta, p})
        + V \sum_{\sigma, \sigma'} \sum_{\braket{pq}} n_{\sigma, p} n_{\sigma', q}

    where :math:`\braket{\dots}` denotes nearest-neighbor pairs and
    :math:`n_{\sigma, p} = a_{\sigma, p}^\dagger a_{\sigma, p}` is the number operator
    on orbital :math:`p` with spin :math:`\sigma`.

    For the tunneling and nearest-neighbor interaction terms, the summations over
    :math:`\braket{pq}` run along edges in the network. Under open boundary conditions,
    there are :math:`N_x \times N_y` vertices and
    :math:`2 (N_x \times N_y) - (N_x + N_y)` edges connected on a plane. Under periodic
    boundary conditions, there are :math:`N_x \times N_y` vertices and
    :math:`2 (N_x \times N_y)` edges connected on a torus, with operator positions
    defined modulo :math:`N_x` or :math:`N_y`. This is a two-dimensional generalization
    of the nearest-neighbor summations defined in
    :func:`fermi_hubbard_1d <~operators.fermi_hubbard.fermi_hubbard_1d>`.

    References:
        - `The Hubbard Model`_

    Args:
        norb_x: The number of spatial orbitals in the x-direction :math:`N_x`.
        norb_y: The number of spatial orbitals in the y-direction :math:`N_y`.
        tunneling: The tunneling amplitude :math:`t`.
        interaction: The onsite interaction strength :math:`U`.
        chemical_potential: The chemical potential :math:`\mu`.
        nearest_neighbor_interaction: The nearest-neighbor interaction strength
            :math:`V`.
        periodic: Whether to use periodic boundary conditions.

    Returns:
        The two-dimensional Fermi-Hubbard model Hamiltonian.

    .. _The Hubbard Model: https://doi.org/10.1146/annurev-conmatphys-031620-102024
    """
    coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = defaultdict(float)

    for orb in range(norb_x * norb_y):
        # Get x and y coordinates of this orbital
        y, x = divmod(orb, norb_x)

        # Get the orbitals to the right and up
        x_right = (x + 1) % norb_x
        y_up = (y + 1) % norb_y
        orb_right = norb_x * y + x_right
        orb_up = norb_x * y_up + x

        if tunneling:
            if x != norb_x - 1 or periodic:
                coeffs[cre_a(orb), des_a(orb_right)] -= tunneling
                coeffs[cre_a(orb_right), des_a(orb)] -= tunneling
                coeffs[cre_b(orb), des_b(orb_right)] -= tunneling
                coeffs[cre_b(orb_right), des_b(orb)] -= tunneling
            if y != norb_y - 1 or periodic:
                coeffs[cre_a(orb), des_a(orb_up)] -= tunneling
                coeffs[cre_a(orb_up), des_a(orb)] -= tunneling
                coeffs[cre_b(orb), des_b(orb_up)] -= tunneling
                coeffs[cre_b(orb_up), des_b(orb)] -= tunneling
        if nearest_neighbor_interaction:
            if x != norb_x - 1 or periodic:
                coeffs[cre_a(orb), des_a(orb), cre_a(orb_right), des_a(orb_right)] += (
                    nearest_neighbor_interaction
                )
                coeffs[cre_a(orb), des_a(orb), cre_b(orb_right), des_b(orb_right)] += (
                    nearest_neighbor_interaction
                )
                coeffs[cre_b(orb), des_b(orb), cre_a(orb_right), des_a(orb_right)] += (
                    nearest_neighbor_interaction
                )
                coeffs[cre_b(orb), des_b(orb), cre_b(orb_right), des_b(orb_right)] += (
                    nearest_neighbor_interaction
                )
            if y != norb_y - 1 or periodic:
                coeffs[cre_a(orb), des_a(orb), cre_a(orb_up), des_a(orb_up)] += (
                    nearest_neighbor_interaction
                )
                coeffs[cre_a(orb), des_a(orb), cre_b(orb_up), des_b(orb_up)] += (
                    nearest_neighbor_interaction
                )
                coeffs[cre_b(orb), des_b(orb), cre_a(orb_up), des_a(orb_up)] += (
                    nearest_neighbor_interaction
                )
                coeffs[cre_b(orb), des_b(orb), cre_b(orb_up), des_b(orb_up)] += (
                    nearest_neighbor_interaction
                )
        if interaction:
            coeffs[cre_a(orb), des_a(orb), cre_b(orb), des_b(orb)] = interaction
        if chemical_potential:
            coeffs[cre_a(orb), des_a(orb)] = -chemical_potential
            coeffs[cre_b(orb), des_b(orb)] = -chemical_potential

    return FermionOperator(coeffs)
