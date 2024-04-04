# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for common operators."""

from __future__ import annotations

import pytest

import ffsim


@pytest.mark.parametrize("orb", range(4))
def test_number_operator(orb: int):
    op = ffsim.number_operator(orb)
    assert dict(op) == {
        (ffsim.cre_a(orb), ffsim.des_a(orb)): 1,
        (ffsim.cre_b(orb), ffsim.des_b(orb)): 1,
    }

    op = ffsim.number_operator(orb, spin=ffsim.Spin.ALPHA)
    assert dict(op) == {
        (ffsim.cre_a(orb), ffsim.des_a(orb)): 1,
    }

    op = ffsim.number_operator(orb, spin=ffsim.Spin.BETA)
    assert dict(op) == {
        (ffsim.cre_b(orb), ffsim.des_b(orb)): 1,
    }
