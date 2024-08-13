# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for FermionOperator."""

from __future__ import annotations

import numpy as np
import pytest

import ffsim
from ffsim import FermionOperator


def test_add():
    """Test adding FermionOperators."""
    op1 = FermionOperator(
        {(ffsim.des_a(1), ffsim.des_a(3), ffsim.cre_a(2), ffsim.cre_a(1)): 1.5}
    )
    op2 = FermionOperator(
        {
            (ffsim.des_a(1), ffsim.cre_b(2), ffsim.des_a(3), ffsim.cre_a(4)): 1,
            (ffsim.des_a(1), ffsim.des_b(3), ffsim.cre_a(2), ffsim.cre_a(4)): 1.5 + 1j,
        }
    )
    expected = FermionOperator(
        {
            (ffsim.des_a(1), ffsim.des_a(3), ffsim.cre_a(2), ffsim.cre_a(1)): 1.5,
            (ffsim.des_a(1), ffsim.cre_b(2), ffsim.des_a(3), ffsim.cre_a(4)): 1,
            (ffsim.des_a(1), ffsim.des_b(3), ffsim.cre_a(2), ffsim.cre_a(4)): 1.5 + 1j,
        }
    )
    assert op1 + op2 == expected

    op1 += op2
    assert op1 == expected


def test_subtract():
    """Test subtracting FermionOperators."""
    op1 = FermionOperator(
        {(ffsim.des_a(1), ffsim.des_a(3), ffsim.cre_a(2), ffsim.cre_a(1)): 1.5}
    )
    op2 = FermionOperator(
        {
            (ffsim.des_a(1), ffsim.cre_b(2), ffsim.des_a(3), ffsim.cre_a(4)): 1,
            (ffsim.des_a(1), ffsim.des_b(3), ffsim.cre_a(2), ffsim.cre_a(4)): 1.5 + 1j,
        }
    )
    expected = FermionOperator(
        {
            (ffsim.des_a(1), ffsim.des_a(3), ffsim.cre_a(2), ffsim.cre_a(1)): 1.5,
            (ffsim.des_a(1), ffsim.cre_b(2), ffsim.des_a(3), ffsim.cre_a(4)): -1,
            (ffsim.des_a(1), ffsim.des_b(3), ffsim.cre_a(2), ffsim.cre_a(4)): -1.5 - 1j,
        }
    )
    assert op1 - op2 == expected

    op1 -= op2
    assert op1 == expected


def test_neg():
    """Test negating FermionOperators."""
    op = FermionOperator(
        {
            (ffsim.des_a(1), ffsim.cre_a(2), ffsim.des_a(3), ffsim.cre_a(4)): 1.5,
            (ffsim.des_a(1), ffsim.des_a(3), ffsim.cre_a(2), ffsim.cre_a(4)): 1 + 1.5j,
        }
    )
    expected = FermionOperator(
        {
            (ffsim.des_a(1), ffsim.cre_a(2), ffsim.des_a(3), ffsim.cre_a(4)): -1.5,
            (ffsim.des_a(1), ffsim.des_a(3), ffsim.cre_a(2), ffsim.cre_a(4)): -1 - 1.5j,
        }
    )
    assert -op == expected


def test_mul():
    """Test multiplying FermionOperators."""
    op1 = FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(5), ffsim.cre_a(3)): 0.5,
            (ffsim.des_a(3), ffsim.cre_b(7), ffsim.cre_b(1), ffsim.cre_a(5)): 0.75,
        }
    )
    op2 = FermionOperator(
        {
            (ffsim.cre_a(2), ffsim.cre_a(5)): 0.5,
            (ffsim.des_a(7),): 0.5,
        }
    )
    expected = FermionOperator(
        {
            (
                ffsim.cre_a(1),
                ffsim.des_a(5),
                ffsim.cre_a(3),
                ffsim.cre_a(2),
                ffsim.cre_a(5),
            ): 0.25,
            (
                ffsim.cre_a(1),
                ffsim.des_a(5),
                ffsim.cre_a(3),
                ffsim.des_a(7),
            ): 0.25,
            (
                ffsim.des_a(3),
                ffsim.cre_b(7),
                ffsim.cre_b(1),
                ffsim.cre_a(5),
                ffsim.cre_a(2),
                ffsim.cre_a(5),
            ): 0.375,
            (
                ffsim.des_a(3),
                ffsim.cre_b(7),
                ffsim.cre_b(1),
                ffsim.cre_a(5),
                ffsim.des_a(7),
            ): 0.375,
        }
    )
    assert op1 * op2 == expected


def test_mul_scalar():
    """Test multiplying by a scalar."""
    op = 1j * FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(5), ffsim.cre_a(3)): 0.5,
            (ffsim.des_a(3), ffsim.cre_b(7), ffsim.cre_b(1), ffsim.cre_a(5)): 0.75,
        }
    )
    expected = FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(5), ffsim.cre_a(3)): 0.5j,
            (ffsim.des_a(3), ffsim.cre_b(7), ffsim.cre_b(1), ffsim.cre_a(5)): 0.75j,
        }
    )
    assert op == expected

    op = 2 * FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(5), ffsim.cre_a(3)): 0.5,
            (ffsim.des_a(3), ffsim.cre_b(7), ffsim.cre_b(1), ffsim.cre_a(5)): 0.75,
        }
    )
    expected = FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(5), ffsim.cre_a(3)): 1,
            (ffsim.des_a(3), ffsim.cre_b(7), ffsim.cre_b(1), ffsim.cre_a(5)): 1.5,
        }
    )
    assert op == expected

    op *= 2
    expected = FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(5), ffsim.cre_a(3)): 2,
            (ffsim.des_a(3), ffsim.cre_b(7), ffsim.cre_b(1), ffsim.cre_a(5)): 3,
        }
    )
    assert op == expected


def test_div():
    """Test division."""
    op = (
        FermionOperator(
            {
                (ffsim.cre_a(1), ffsim.des_a(5), ffsim.cre_a(3)): 0.5,
                (ffsim.des_a(3), ffsim.cre_b(7), ffsim.cre_b(1), ffsim.cre_a(5)): 0.75,
            }
        )
        / 2
    )
    expected = FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(5), ffsim.cre_a(3)): 0.25,
            (ffsim.des_a(3), ffsim.cre_b(7), ffsim.cre_b(1), ffsim.cre_a(5)): 0.375,
        }
    )
    assert op == expected

    op /= 2
    expected = FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(5), ffsim.cre_a(3)): 0.125,
            (ffsim.des_a(3), ffsim.cre_b(7), ffsim.cre_b(1), ffsim.cre_a(5)): 0.1875,
        }
    )


def test_pow():
    """Test exponentiation by an integer."""
    op = FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(5), ffsim.cre_a(3)): 0.5,
            (ffsim.des_a(3), ffsim.cre_b(7), ffsim.cre_b(1), ffsim.cre_a(5)): 0.75,
        }
    )
    assert op**0 == FermionOperator({(): 1})
    assert op**1 == op
    assert op**2 == op * op
    assert op**3 == op * op * op
    assert pow(op, 2) == op * op
    with pytest.raises(ValueError, match="mod argument"):
        _ = pow(op, 2, 2)


def test_normal_ordered():
    """Test normal ordering."""
    actual = FermionOperator(
        {
            (
                ffsim.des_a(1),
                ffsim.des_b(1),
                ffsim.cre_b(2),
                ffsim.cre_a(2),
                ffsim.des_a(3),
                ffsim.des_b(3),
                ffsim.cre_b(4),
                ffsim.cre_a(4),
            ): 1.5
        }
    ).normal_ordered()
    expected = FermionOperator(
        {
            (
                ffsim.cre_b(4),
                ffsim.cre_b(2),
                ffsim.cre_a(4),
                ffsim.cre_a(2),
                ffsim.des_b(3),
                ffsim.des_b(1),
                ffsim.des_a(3),
                ffsim.des_a(1),
            ): 1.5
        }
    )
    assert actual == expected
    for term in actual:
        assert list(term) == sorted(term, reverse=True)

    actual = FermionOperator(
        {(ffsim.des_a(1), ffsim.des_a(3), ffsim.cre_a(2), ffsim.cre_a(1)): 1.5}
    ).normal_ordered()
    expected = FermionOperator(
        {
            (ffsim.cre_a(2), ffsim.cre_a(1), ffsim.des_a(3), ffsim.des_a(1)): -1.5,
            (ffsim.cre_a(2), ffsim.des_a(3)): -1.5,
        }
    )
    assert actual == expected
    for term in actual:
        assert list(term) == sorted(term, reverse=True)

    actual = FermionOperator(
        {
            (ffsim.des_a(1), ffsim.cre_a(2), ffsim.des_a(3), ffsim.cre_a(4)): 1.5,
            (ffsim.des_a(1), ffsim.des_a(3), ffsim.cre_a(2), ffsim.cre_a(1)): 1.5,
        }
    ).normal_ordered()
    expected = FermionOperator(
        {
            (ffsim.cre_a(4), ffsim.cre_a(2), ffsim.des_a(3), ffsim.des_a(1)): -1.5,
            (ffsim.cre_a(2), ffsim.cre_a(1), ffsim.des_a(3), ffsim.des_a(1)): -1.5,
            (ffsim.cre_a(2), ffsim.des_a(3)): -1.5,
        }
    )
    assert actual == expected
    for term in actual:
        assert list(term) == sorted(term, reverse=True)

    actual = FermionOperator(
        {
            (
                ffsim.cre_a(1),
                ffsim.des_a(3),
                ffsim.des_a(5),
                ffsim.cre_a(7),
                ffsim.cre_a(3),
            ): 0.5,
            (
                ffsim.des_a(3),
                ffsim.des_a(5),
                ffsim.cre_a(7),
                ffsim.cre_a(1),
                ffsim.cre_a(5),
                ffsim.cre_a(3),
            ): 0.5,
            (
                ffsim.des_a(3),
                ffsim.des_a(5),
                ffsim.cre_a(7),
                ffsim.cre_a(3),
                ffsim.cre_a(1),
            ): 0.75,
        }
    ).normal_ordered()
    expected = FermionOperator(
        {
            (ffsim.cre_a(7), ffsim.cre_a(1)): 0.5,
            (ffsim.cre_a(7), ffsim.cre_a(1), ffsim.des_a(5)): 1.25,
            (ffsim.cre_a(7), ffsim.cre_a(3), ffsim.cre_a(1), ffsim.des_a(3)): 0.5,
            (
                ffsim.cre_a(7),
                ffsim.cre_a(3),
                ffsim.cre_a(1),
                ffsim.des_a(5),
                ffsim.des_a(3),
            ): -1.25,
            (ffsim.cre_a(7), ffsim.cre_a(5), ffsim.cre_a(1), ffsim.des_a(5)): 0.5,
            (
                ffsim.cre_a(7),
                ffsim.cre_a(5),
                ffsim.cre_a(3),
                ffsim.cre_a(1),
                ffsim.des_a(5),
                ffsim.des_a(3),
            ): -0.5,
        }
    )
    assert actual == expected
    for term in actual:
        assert list(term) == sorted(term, reverse=True)


def test_conserves_particle_number():
    op = FermionOperator(
        {
            (
                ffsim.cre_a(1),
                ffsim.des_a(3),
                ffsim.des_a(7),
                ffsim.cre_a(3),
            ): 0.5,
            (ffsim.des_a(3), ffsim.cre_a(7)): 0.75,
        }
    )
    assert op.conserves_particle_number()

    op = FermionOperator(
        {
            (
                ffsim.cre_a(1),
                ffsim.des_a(3),
                ffsim.des_a(7),
                ffsim.cre_a(3),
            ): 0.5,
            (ffsim.des_a(3), ffsim.cre_a(7)): 0.75,
            (
                ffsim.cre_a(1),
                ffsim.des_a(3),
                ffsim.des_a(5),
                ffsim.cre_a(7),
                ffsim.cre_a(3),
            ): 0.5,
        }
    )
    assert not op.conserves_particle_number()


def test_conserves_spin_z():
    op = FermionOperator(
        {
            (
                ffsim.cre_a(1),
                ffsim.cre_a(2),
                ffsim.des_a(3),
                ffsim.des_b(1),
                ffsim.cre_b(2),
                ffsim.cre_b(3),
            ): 0.5,
            (
                ffsim.des_a(3),
                ffsim.des_b(7),
            ): 0.75,
        }
    )
    assert op.conserves_spin_z()

    op = FermionOperator(
        {
            (
                ffsim.cre_a(1),
                ffsim.cre_a(2),
                ffsim.des_a(3),
                ffsim.des_b(1),
                ffsim.cre_b(2),
                ffsim.cre_b(3),
            ): 0.5,
            (
                ffsim.des_a(3),
                ffsim.cre_b(7),
            ): 0.75,
        }
    )
    assert not op.conserves_spin_z()


def test_many_body_order():
    op = FermionOperator({})
    assert op.many_body_order() == 0

    op = FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(3)): 0.5,
            (ffsim.des_a(3), ffsim.cre_a(7)): 0.75,
        }
    )
    assert op.many_body_order() == 2

    op = FermionOperator(
        {
            (
                ffsim.cre_a(1),
                ffsim.des_a(3),
                ffsim.des_a(7),
                ffsim.cre_a(3),
            ): 0.5,
            (ffsim.des_a(3), ffsim.cre_a(7)): 0.75,
            (
                ffsim.cre_a(1),
                ffsim.des_a(3),
                ffsim.des_a(5),
                ffsim.cre_a(7),
                ffsim.cre_a(3),
            ): 0.5,
        }
    )
    assert op.many_body_order() == 5


def test_get_set():
    op = FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(3)): 0.5,
            (ffsim.des_a(3), ffsim.cre_a(7)): 0.75,
        }
    )
    assert op[(ffsim.cre_a(1), ffsim.des_a(3))] == 0.5

    op[(ffsim.cre_a(1), ffsim.des_a(3))] = 0.25
    assert op[(ffsim.cre_a(1), ffsim.des_a(3))] == 0.25


def test_del():
    op = FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(3)): 0.5,
            (ffsim.des_a(3), ffsim.cre_a(7)): 0.75,
        }
    )
    assert op[(ffsim.cre_a(1), ffsim.des_a(3))] == 0.5

    del op[(ffsim.cre_a(1), ffsim.des_a(3))]
    assert (ffsim.cre_a(1), ffsim.des_a(3)) not in op


def test_len():
    op = FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(3)): 0.5,
            (ffsim.des_a(3), ffsim.cre_a(7)): 0.75,
        }
    )
    assert len(op) == 2


def test_iter():
    op = FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(3)): 0.5,
            (ffsim.des_a(3), ffsim.cre_a(7)): 0.75,
            (ffsim.des_a(3), ffsim.cre_a(7), ffsim.cre_b(6)): 0.75,
        }
    )
    assert set(op) == {
        (ffsim.cre_a(1), ffsim.des_a(3)),
        (ffsim.des_a(3), ffsim.cre_a(7)),
        (ffsim.des_a(3), ffsim.cre_a(7), ffsim.cre_b(6)),
    }


def test_linear_operator_one_body():
    """Test linear operator of a one-body operator."""
    norb = 5

    rng = np.random.default_rng()

    one_body_tensor = np.zeros((norb, norb)).astype(complex)
    one_body_tensor[1, 2] = 0.5
    one_body_tensor[2, 1] = -0.5
    one_body_tensor[3, 1] = 1 + 1j

    for nelec in [(2, 2), (0, 2), (5, 4)]:
        expected_linop = ffsim.contract.one_body_linop(
            one_body_tensor, norb=norb, nelec=nelec
        )

        op = FermionOperator(
            {
                (ffsim.cre_a(1), ffsim.des_a(2)): 0.5,
                (ffsim.cre_a(2), ffsim.des_a(1)): -0.5,
                (ffsim.cre_a(3), ffsim.des_a(1)): 1 + 1j,
                (ffsim.cre_b(1), ffsim.des_b(2)): 0.5,
                (ffsim.cre_b(2), ffsim.des_b(1)): -0.5,
                (ffsim.cre_b(3), ffsim.des_b(1)): 1 + 1j,
            }
        )
        actual_linop = ffsim.linear_operator(op, norb, nelec)
        vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=rng)
        original = vec.copy()
        actual = actual_linop @ vec
        expected = expected_linop @ vec
        # test no side effect
        np.testing.assert_allclose(original, vec)
        # check results match
        np.testing.assert_allclose(actual, expected)

        op = FermionOperator(
            {
                (ffsim.des_a(2), ffsim.cre_a(1)): -0.5,
                (ffsim.des_a(1), ffsim.cre_a(2)): 0.5,
                (ffsim.cre_a(3), ffsim.des_a(1)): 1 + 1j,
                (ffsim.des_b(2), ffsim.cre_b(1)): -0.5,
                (ffsim.des_b(1), ffsim.cre_b(2)): 0.5,
                (ffsim.cre_b(3), ffsim.des_b(1)): 1 + 1j,
            }
        )
        actual_linop = ffsim.linear_operator(op, norb, nelec)
        vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=rng)
        original = vec.copy()
        actual = actual_linop @ vec
        expected = expected_linop @ vec
        # test no side effect
        np.testing.assert_allclose(original, vec)
        # check results match
        np.testing.assert_allclose(actual, expected)


def test_approx_eq():
    """Test approximate equality."""
    op1 = FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(2)): 0.5,
            (ffsim.cre_a(2), ffsim.des_a(1)): -0.5,
            (ffsim.cre_b(1), ffsim.des_b(2)): 0.5,
            (ffsim.cre_b(2), ffsim.des_b(1)): -0.5,
        }
    )
    op2 = FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(2)): 0.5 + 1e-7,
            (ffsim.cre_a(2), ffsim.des_a(1)): -0.5 - 1e-7,
            (ffsim.cre_b(1), ffsim.des_b(2)): 0.5,
            (ffsim.cre_b(2), ffsim.des_b(1)): -0.5,
        }
    )
    assert ffsim.approx_eq(op1, op2)
    assert ffsim.approx_eq(op1, op2, rtol=0, atol=1e-7)
    assert not ffsim.approx_eq(op1, op2, rtol=0)


def test_repr_equivalent():
    """Test that repr evaluates to an equivalent object."""
    op = FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(2)): 1,
            (ffsim.cre_a(2), ffsim.des_a(1)): 0.5,
            (ffsim.cre_b(1), ffsim.des_b(2)): -0.5j,
            (ffsim.cre_b(2), ffsim.des_b(1)): 1 - 0.5j,
        }
    )
    assert eval(repr(op)) == op


def test_str_equivalent():
    """Test that str evaluates to an equivalent object."""
    op = FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(2)): 1,
            (ffsim.cre_a(2), ffsim.des_a(1)): 0.5,
            (ffsim.cre_b(1), ffsim.des_b(2)): -0.5j,
            (ffsim.cre_b(2), ffsim.des_b(1)): 1 - 0.5j,
        }
    )
    exec("from ffsim import cre_a, cre_b, des_a, des_b")
    assert eval(str(op)) == op


def test_copy():
    op = FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(2)): 1,
            (ffsim.cre_a(2), ffsim.des_a(1)): 0.5,
            (ffsim.cre_b(1), ffsim.des_b(2)): -0.5j,
            (ffsim.cre_b(2), ffsim.des_b(1)): 1 - 0.5j,
        }
    )
    copy = op.copy()
    assert copy == op

    copy *= 2
    assert copy != op


def test_mapping_methods():
    op = FermionOperator(
        {
            (ffsim.cre_a(1), ffsim.des_a(2)): 1,
            (ffsim.cre_a(2), ffsim.des_a(1)): 0.5,
            (ffsim.cre_b(1), ffsim.des_b(2)): -0.5j,
            (ffsim.cre_b(2), ffsim.des_b(1)): 1 - 0.5j,
        }
    )
    assert op.keys() == {
        (ffsim.cre_a(1), ffsim.des_a(2)),
        (ffsim.cre_a(2), ffsim.des_a(1)),
        (ffsim.cre_b(1), ffsim.des_b(2)),
        (ffsim.cre_b(2), ffsim.des_b(1)),
    }
    assert set(op.values()) == {
        1,
        0.5,
        -0.5j,
        1 - 0.5j,
    }
    assert op.items() == {
        ((ffsim.cre_a(1), ffsim.des_a(2)), 1),
        ((ffsim.cre_a(2), ffsim.des_a(1)), 0.5),
        ((ffsim.cre_b(1), ffsim.des_b(2)), -0.5j),
        ((ffsim.cre_b(2), ffsim.des_b(1)), 1 - 0.5j),
    }
