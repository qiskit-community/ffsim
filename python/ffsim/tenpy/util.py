# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""TeNPy utility functions."""

from __future__ import annotations

import tenpy.linalg.np_conserved as npc
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfFermionSite


def bitstring_to_mps(bitstring: tuple[int, int], norb: int) -> MPS:
    r"""Return the bitstring as an MPS.

    Args:
        bitstring: The bitstring in the form `(int_a, int_b)`.
        norb: The number of spatial orbitals.

    Returns:
        The bitstring as an MPS.
    """

    # unpack bitstrings
    int_a, int_b = bitstring
    string_a = format(int_a, f"0{norb}b")
    string_b = format(int_b, f"0{norb}b")

    # relabel using TeNPy SpinHalfFermionSite convention
    product_state = []
    swap_factor = 1
    occupation = {"00": 0, "10": 1, "01": 2, "11": 3}
    previous_site_occupation = None
    for i, site in enumerate(zip(reversed(string_a), reversed(string_b))):
        site_occupation = occupation["".join(site)]
        product_state.append(site_occupation)

        if site_occupation in [1, 3] and previous_site_occupation in [2, 3]:
            swap_factor *= -1

        previous_site_occupation = site_occupation

    # construct product state MPS
    shfs = SpinHalfFermionSite(cons_N="N", cons_Sz="Sz")
    mps = MPS.from_product_state([shfs] * norb, product_state)

    # map from ffsim to TeNPy ordering
    if swap_factor == -1:
        minus_identity_npc = npc.Array.from_ndarray(
            -shfs.get_op("Id").to_ndarray(),
            [shfs.leg, shfs.leg.conj()],
            labels=["p", "p*"],
        )
        mps.apply_local_op(0, minus_identity_npc)

    return mps
