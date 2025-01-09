# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import pytest
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfFermionSite

from ffsim.tenpy.util import bitstring_to_mps


@pytest.mark.parametrize(
    "bitstring, norb, product_state",
    [
        ((0, 0), 2, [0, 0]),
        ((2, 0), 2, [0, 1]),
        ((0, 2), 2, [0, 2]),
        ((2, 2), 2, [0, 3]),
        ((1, 0), 2, [1, 0]),
        ((3, 0), 2, [1, 1]),
        ((1, 2), 2, [1, 2]),
        ((3, 2), 2, [1, 3]),
        ((0, 1), 2, [2, 0]),
        ((2, 1), 2, [2, 1]),
        ((0, 3), 2, [2, 2]),
        ((2, 3), 2, [2, 3]),
        ((1, 1), 2, [3, 0]),
        ((3, 1), 2, [3, 1]),
        ((1, 3), 2, [3, 2]),
        ((3, 3), 2, [3, 3]),
    ],
)
def test_bitstring_to_mps(bitstring: tuple[int, int], norb: int, product_state: list):
    """Test converting a bitstring to an MPS."""

    # convert bitstring to MPS
    mps = bitstring_to_mps(bitstring, norb)

    # construct expected MPS
    shfs = SpinHalfFermionSite(cons_N="N", cons_Sz="Sz")
    expected_mps = MPS.from_product_state([shfs] * norb, product_state)

    # map from ffsim to TeNPy ordering
    if product_state[0] in [2, 3] and product_state[1] in [1, 3]:
        minus_identity_npc = npc.Array.from_ndarray(
            -shfs.get_op("Id").to_ndarray(),
            [shfs.leg, shfs.leg.conj()],
            labels=["p", "p*"],
        )
        expected_mps.apply_local_op(0, minus_identity_npc)

    # test overlap is one
    overlap = mps.overlap(expected_mps)
    np.testing.assert_equal(overlap, 1)
