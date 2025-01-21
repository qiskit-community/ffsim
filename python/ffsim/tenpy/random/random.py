# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from tenpy.algorithms.tebd import RandomUnitaryEvolution
from tenpy.networks.mps import MPS

import ffsim
from ffsim.tenpy.util import bitstring_to_mps


def random_mps(
    norb: int, nelec: tuple[int, int], n_steps: int = 10, chi_max: int = 100
) -> MPS:
    """Return a random MPS generated from a random unitary evolution.

    Args:
        norb: The number of orbitals.
        nelec: The number of electrons.
        n_steps: The number of steps in the random unitary evolution.
        chi_max: The maximum bond dimension in the random unitary evolution.

    Returns:
        The random MPS.
    """

    # initialize Hartree-Fock state
    dim = ffsim.dim(norb, nelec)
    strings = ffsim.addresses_to_strings(
        range(dim), norb=norb, nelec=nelec, bitstring_type=ffsim.BitstringType.STRING
    )
    string_tuples = [
        (
            int(string[len(string) // 2 :], base=2),
            int(string[: len(string) // 2], base=2),
        )
        for string in strings
    ]
    mps = bitstring_to_mps(string_tuples[0], norb)

    # apply random unitary evolution
    tebd_params = {
        "N_steps": n_steps,
        "trunc_params": {"chi_max": chi_max},
        "verbose": 0,
    }
    eng = RandomUnitaryEvolution(mps, tebd_params)
    eng.run()
    mps.canonical_form()

    return mps
