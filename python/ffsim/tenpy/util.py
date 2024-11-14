# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfFermionSite


def product_state_as_mps(bitstring: tuple[str, str]) -> MPS:
    r"""Return the product state as an MPS.

    Args:
        bitstring: The bitstring in the form `(string_a, string_b)`.

    Returns:
        The product state as an MPS.
    """

    # unpack bitstrings
    string_a, string_b = bitstring

    # extract norb
    assert len(string_a) == len(string_b)
    norb = len(string_a)

    # merge bitstrings
    up_sector = string_a.replace("1", "u")
    down_sector = string_b.replace("1", "d")
    product_state = [a + b for a, b in zip(up_sector, down_sector)]

    # relabel using TeNPy SpinHalfFermionSite convention
    for i, site in enumerate(product_state):
        if site == "00":
            product_state[i] = "empty"
        elif site == "u0":
            product_state[i] = "up"
        elif site == "0d":
            product_state[i] = "down"
        elif site == "ud":
            product_state[i] = "full"
        else:
            raise ValueError("undefined site")

    # note that the bit positions increase from right to left
    product_state = product_state[::-1]

    # construct product state MPS
    shfs = SpinHalfFermionSite(cons_N="N", cons_Sz="Sz")
    psi_mps = MPS.from_product_state([shfs] * norb, product_state)

    return psi_mps
