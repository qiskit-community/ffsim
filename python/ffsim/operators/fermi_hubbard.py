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

        H = -t \sum_{\sigma} \sum_{p=1}^{N-1}
        (a^\dagger_{\sigma, p} a_{\sigma, p+1} + \text{h.c.})
        + U \sum_{p=1}^{N} n_{\alpha, p} n_{\beta, p}
        - \mu \sum_{p=1}^N (n_{\alpha, p} + n_{\beta, p})
        + V \sum_{\sigma, \sigma'} \sum_{p=1}^{N-1} n_{\sigma, p} n_{\sigma', p+1}

    where :math:`n_{\sigma, p} = a_{\sigma, p}^\dagger a_{\sigma, p}` is the number
    operator on orbital :math:`p` with spin :math:`\sigma`.

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
    coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {}

    # initialize tunneling keys
    for p in range(norb - 1 + periodic):
        coeffs[(cre_a(p), des_a((p + 1) % norb))] = 0
        coeffs[(cre_b(p), des_b((p + 1) % norb))] = 0
        coeffs[(cre_a((p + 1) % norb), des_a(p))] = 0
        coeffs[(cre_b((p + 1) % norb), des_b(p))] = 0

    # populate keys
    for p in range(norb - 1 + periodic):
        coeffs[(cre_a(p), des_a((p + 1) % norb))] -= tunneling
        coeffs[(cre_b(p), des_b((p + 1) % norb))] -= tunneling
        coeffs[(cre_a((p + 1) % norb), des_a(p))] -= tunneling
        coeffs[(cre_b((p + 1) % norb), des_b(p))] -= tunneling
        if nearest_neighbor_interaction:
            coeffs[
                (cre_a(p), des_a(p), cre_a((p + 1) % norb), des_a((p + 1) % norb))
            ] = nearest_neighbor_interaction
            coeffs[
                (cre_a(p), des_a(p), cre_b((p + 1) % norb), des_b((p + 1) % norb))
            ] = nearest_neighbor_interaction
            coeffs[
                (cre_b(p), des_b(p), cre_a((p + 1) % norb), des_a((p + 1) % norb))
            ] = nearest_neighbor_interaction
            coeffs[
                (cre_b(p), des_b(p), cre_b((p + 1) % norb), des_b((p + 1) % norb))
            ] = nearest_neighbor_interaction

    for p in range(norb):
        if interaction:
            coeffs[(cre_a(p), des_a(p), cre_b(p), des_b(p))] = interaction
        if chemical_potential:
            coeffs[(cre_a(p), des_a(p))] = -chemical_potential
            coeffs[(cre_b(p), des_b(p))] = -chemical_potential

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
    coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {}

    # initialize tunneling keys
    for x in range(norb_x):
        for y in range(norb_y):
            # position in Cartesian coordinates
            x_right = (x + 1) % norb_x
            y_up = (y + 1) % norb_y

            # position on C-style chain
            p = norb_y * x + y
            p_right = norb_y * x_right + y
            p_up = norb_y * x + y_up

            if x != norb_x - 1 or periodic:
                coeffs[(cre_a(p), des_a(p_right))] = 0
                coeffs[(cre_a(p_right), des_a(p))] = 0
                coeffs[(cre_b(p), des_b(p_right))] = 0
                coeffs[(cre_b(p_right), des_b(p))] = 0

            if y != norb_y - 1 or periodic:
                coeffs[(cre_a(p), des_a(p_up))] = 0
                coeffs[(cre_a(p_up), des_a(p))] = 0
                coeffs[(cre_b(p), des_b(p_up))] = 0
                coeffs[(cre_b(p_up), des_b(p))] = 0

    # populate keys
    for x in range(norb_x):
        for y in range(norb_y):
            # position in Cartesian coordinates
            x_right = (x + 1) % norb_x
            y_up = (y + 1) % norb_y

            # position on C-style chain
            p = norb_y * x + y
            p_right = norb_y * x_right + y
            p_up = norb_y * x + y_up

            if x != norb_x - 1 or periodic:
                coeffs[(cre_a(p), des_a(p_right))] -= tunneling
                coeffs[(cre_a(p_right), des_a(p))] -= tunneling
                coeffs[(cre_b(p), des_b(p_right))] -= tunneling
                coeffs[(cre_b(p_right), des_b(p))] -= tunneling
                if nearest_neighbor_interaction:
                    coeffs[(cre_a(p), des_a(p), cre_a(p_right), des_a(p_right))] = (
                        nearest_neighbor_interaction
                    )
                    coeffs[(cre_a(p), des_a(p), cre_b(p_right), des_b(p_right))] = (
                        nearest_neighbor_interaction
                    )
                    coeffs[(cre_b(p), des_b(p), cre_a(p_right), des_a(p_right))] = (
                        nearest_neighbor_interaction
                    )
                    coeffs[(cre_b(p), des_b(p), cre_b(p_right), des_b(p_right))] = (
                        nearest_neighbor_interaction
                    )

            if y != norb_y - 1 or periodic:
                coeffs[(cre_a(p), des_a(p_up))] -= tunneling
                coeffs[(cre_a(p_up), des_a(p))] -= tunneling
                coeffs[(cre_b(p), des_b(p_up))] -= tunneling
                coeffs[(cre_b(p_up), des_b(p))] -= tunneling
                if nearest_neighbor_interaction:
                    coeffs[(cre_a(p), des_a(p), cre_a(p_up), des_a(p_up))] = (
                        nearest_neighbor_interaction
                    )
                    coeffs[(cre_a(p), des_a(p), cre_b(p_up), des_b(p_up))] = (
                        nearest_neighbor_interaction
                    )
                    coeffs[(cre_b(p), des_b(p), cre_a(p_up), des_a(p_up))] = (
                        nearest_neighbor_interaction
                    )
                    coeffs[(cre_b(p), des_b(p), cre_b(p_up), des_b(p_up))] = (
                        nearest_neighbor_interaction
                    )

    for p in range(norb_x * norb_y):
        if interaction:
            coeffs[(cre_a(p), des_a(p), cre_b(p), des_b(p))] = interaction
        if chemical_potential:
            coeffs[(cre_a(p), des_a(p))] = -chemical_potential
            coeffs[(cre_b(p), des_b(p))] = -chemical_potential

    return FermionOperator(coeffs)
