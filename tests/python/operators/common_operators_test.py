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


def equivalent(op1: ffsim.FermionOperator, op2: ffsim.FermionOperator) -> bool:
    """Return whether two FermionOperators represent the same operator."""
    diff = (op1 - op2).normal_ordered()
    diff.simplify(tol=0)
    return not diff


@pytest.mark.parametrize("norb", range(5))
def test_s_plus_operator(norb: int):
    op = ffsim.s_plus_operator(norb)
    assert dict(op) == {(ffsim.cre_a(i), ffsim.des_b(i)): 1 for i in range(norb)}


@pytest.mark.parametrize("norb", range(5))
def test_s_minus_operator(norb: int):
    op = ffsim.s_minus_operator(norb)
    assert dict(op) == {(ffsim.cre_b(i), ffsim.des_a(i)): 1 for i in range(norb)}


@pytest.mark.parametrize("norb", range(5))
def test_sx_operator(norb: int):
    op = ffsim.s_x_operator(norb)
    expected = 0.5 * (ffsim.s_plus_operator(norb) + ffsim.s_minus_operator(norb))
    assert op == expected


@pytest.mark.parametrize("norb", range(5))
def test_sy_operator(norb: int):
    op = ffsim.s_y_operator(norb)
    expected = (-0.5j) * (ffsim.s_plus_operator(norb) - ffsim.s_minus_operator(norb))
    assert op == expected


@pytest.mark.parametrize("norb", range(5))
def test_sz_operator(norb: int):
    op = ffsim.s_z_operator(norb)
    assert dict(op) == {
        **{(ffsim.cre_a(i), ffsim.des_a(i)): 0.5 for i in range(norb)},
        **{(ffsim.cre_b(i), ffsim.des_b(i)): -0.5 for i in range(norb)},
    }


@pytest.mark.parametrize("norb", range(5))
def test_s_squared_equals_sum_of_squares(norb: int):
    """S^2 = Sx^2 + Sy^2 + Sz^2."""
    assert equivalent(
        ffsim.s_squared_operator(norb),
        ffsim.s_x_operator(norb) ** 2
        + ffsim.s_y_operator(norb) ** 2
        + ffsim.s_z_operator(norb) ** 2,
    )


@pytest.mark.parametrize("norb", range(5))
def test_s_plus_equals_sx_plus_i_sy(norb: int):
    """S+ = Sx + i*Sy."""
    assert equivalent(
        ffsim.s_plus_operator(norb),
        ffsim.s_x_operator(norb) + 1j * ffsim.s_y_operator(norb),
    )


@pytest.mark.parametrize("norb", range(5))
def test_s_minus_equals_sx_minus_i_sy(norb: int):
    """S- = Sx - i*Sy."""
    assert equivalent(
        ffsim.s_minus_operator(norb),
        ffsim.s_x_operator(norb) - 1j * ffsim.s_y_operator(norb),
    )


@pytest.mark.parametrize("norb", range(5))
def test_spin_commutation_relations(norb: int):
    """[Sx, Sy] = i*Sz, [Sy, Sz] = i*Sx, [Sz, Sx] = i*Sy."""
    sx = ffsim.s_x_operator(norb)
    sy = ffsim.s_y_operator(norb)
    sz = ffsim.s_z_operator(norb)
    for a, b, c in [(sx, sy, sz), (sy, sz, sx), (sz, sx, sy)]:
        assert equivalent(a * b - b * a, 1j * c)


@pytest.mark.parametrize("orb", range(5))
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
