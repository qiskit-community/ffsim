# (C) Copyright IBM 2024.
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
    for site in zip(reversed(string_a), reversed(string_b)):
        if site == ("0", "0"):
            product_state.append("empty")
        elif site == ("1", "0"):
            product_state.append("up")
        elif site == ("0", "1"):
            product_state.append("down")
        else:  # site == ("1", "1"):
            product_state.append("full")

    # construct product state MPS
    shfs = SpinHalfFermionSite(cons_N="N", cons_Sz="Sz")
    mps = MPS.from_product_state([shfs] * norb, product_state)

    return mps
