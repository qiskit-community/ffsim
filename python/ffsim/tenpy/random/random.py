# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import random

from tenpy.algorithms.tebd import RandomUnitaryEvolution
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfFermionSite

from ffsim.tenpy.util import _bitstring_to_product_state


def random_mps(
    norb: int, nelec: tuple[int, int], n_steps: int = 10, chi_max: int = 100
) -> MPS:
    """Return a random MPS generated from a random unitary evolution.

    Args:
        norb: The number of spatial orbitals.
        nelec: The number of electrons.
        n_steps: The number of steps in the random unitary evolution.
        chi_max: The maximum bond dimension in the random unitary evolution.

    Returns:
        The random MPS.
    """

    # initialize Hartree-Fock state
    n_alpha, n_beta = nelec
    product_state = _bitstring_to_product_state(
        ((1 << n_alpha) - 1, (1 << n_beta) - 1), norb
    )
    shfs = SpinHalfFermionSite(cons_N="N", cons_Sz="Sz")
    mps = MPS.from_product_state([shfs] * norb, product_state)

    # apply random unitary evolution
    options = {"N_steps": n_steps, "trunc_params": {"chi_max": chi_max}}
    eng = RandomUnitaryEvolution(mps, options)
    eng.run()
    mps.canonical_form()

    return mps


def random_mps_product_state(norb: int, nelec: tuple[int, int]) -> MPS:
    """Return a random MPS product state.

    Args:
        norb: The number of spatial orbitals.
        nelec: The number of electrons.

    Returns:
        The random MPS product state.
    """
    (n_alpha, n_beta) = nelec

    n_alpha_list = [1] * n_alpha + [0] * (norb - n_alpha)
    random.shuffle(n_alpha_list)
    n_beta_list = [1] * n_beta + [0] * (norb - n_beta)
    random.shuffle(n_beta_list)

    s_alpha = sum(j << i for i, j in enumerate(n_alpha_list))
    s_beta = sum(j << i for i, j in enumerate(n_beta_list))
    product_state = _bitstring_to_product_state((s_alpha, s_beta), norb)

    shfs = SpinHalfFermionSite(cons_N="N", cons_Sz="Sz")
    mps = MPS.from_product_state([shfs] * norb, product_state)

    return mps
