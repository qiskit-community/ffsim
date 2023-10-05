# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for FermionAction."""

from __future__ import annotations

import ffsim


def test_fermion_action():
    assert (
        ffsim.cre_a(0)
        == ffsim.cre(False, 0)
        == ffsim.FermionAction(action=True, orb=0, spin=False)
        == (True, False, 0)
    )
    assert (
        ffsim.cre_b(1)
        == ffsim.cre(True, 1)
        == ffsim.FermionAction(action=True, orb=1, spin=True)
        == (True, True, 1)
    )
    assert (
        ffsim.des_a(-1)
        == ffsim.des(False, -1)
        == ffsim.FermionAction(action=False, orb=-1, spin=False)
        == (False, False, -1)
    )
    assert (
        ffsim.des_b(2)
        == ffsim.des(True, 2)
        == ffsim.FermionAction(action=False, orb=2, spin=True)
        == (False, True, 2)
    )
