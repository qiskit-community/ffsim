# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test reduced density matrices."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import ffsim


@pytest.mark.parametrize(
    "norb, nelec, spin_summed",
    [
        (4, (2, 2), True),
        (4, (2, 2), False),
        (3, (1, 2), True),
        (3, (1, 2), False),
        (2, (0, 1), True),
        (2, (0, 1), False),
        (1, (0, 0), True),
        (1, (0, 0), False),
    ],
)
def test_rdm_1(norb: int, nelec: tuple[int, int], spin_summed: bool):
    """Test computing 1-RDM."""
    func = {
        # spin_summed: function
        False: _rdm1,
        True: _rdm1_spin_summed,
    }
    rng = np.random.default_rng()
    vec = ffsim.random.random_statevector(ffsim.dim(norb, nelec), seed=rng)

    rdm = ffsim.rdm(vec, norb, nelec, spin_summed=spin_summed)
    expected = func[spin_summed](vec, norb, nelec)
    np.testing.assert_allclose(rdm, expected, atol=1e-12)


@pytest.mark.parametrize(
    "norb, nelec, spin_summed, reordered",
    [
        (4, (2, 2), True, True),
        (4, (2, 2), True, False),
        (4, (2, 2), False, True),
        (4, (2, 2), False, False),
        (3, (1, 2), True, True),
        (3, (1, 2), True, False),
        (3, (1, 2), False, True),
        (3, (1, 2), False, False),
        (2, (0, 1), True, True),
        (2, (0, 1), True, False),
        (2, (0, 1), False, True),
        (2, (0, 1), False, False),
        (1, (0, 0), True, True),
        (1, (0, 0), True, False),
        (1, (0, 0), False, True),
        (1, (0, 0), False, False),
    ],
)
def test_rdm_2(norb: int, nelec: tuple[int, int], spin_summed: bool, reordered: bool):
    """Test computing 1- and 2-RDMs."""
    func = {
        # (spin_summed, reorder): (rdm1_function, rdm2_function)
        (False, False): (_rdm1, _rdm2),
        (False, True): (_rdm1, _rdm2_reordered),
        (True, False): (_rdm1_spin_summed, _rdm2_spin_summed),
        (True, True): (_rdm1_spin_summed, _rdm2_spin_summed_reordered),
    }
    rng = np.random.default_rng()
    vec = ffsim.random.random_statevector(ffsim.dim(norb, nelec), seed=rng)

    rdm1, rdm2 = ffsim.rdm(
        vec,
        norb,
        nelec,
        rank=2,
        spin_summed=spin_summed,
        reordered=reordered,
    )
    rdm1_func, rdm2_func = func[spin_summed, reordered]
    expected_rdm1 = rdm1_func(vec, norb, nelec)
    expected_rdm2 = rdm2_func(vec, norb, nelec)
    np.testing.assert_allclose(rdm1, expected_rdm1, atol=1e-12)
    np.testing.assert_allclose(rdm2, expected_rdm2, atol=1e-12)


@pytest.mark.parametrize(
    "norb, nelec, spin_summed, reordered, rank",
    [
        (4, (2, 2), True, True, 2),
    ],
)
def test_no_lower_ranks(
    norb: int, nelec: tuple[int, int], spin_summed: bool, reordered: bool, rank: int
):
    """Test computing higher-rank RDM without returning lower ranks."""
    func = {
        # (spin_summed, reorder): function
        (False, False, 2): _rdm2,
        (False, True, 2): _rdm2_reordered,
        (True, False, 2): _rdm2_spin_summed,
        (True, True, 2): _rdm2_spin_summed_reordered,
    }
    rng = np.random.default_rng()
    vec = ffsim.random.random_statevector(ffsim.dim(norb, nelec), seed=rng)

    rdm = ffsim.rdm(
        vec,
        norb,
        nelec,
        rank=rank,
        spin_summed=spin_summed,
        reordered=reordered,
        return_lower_ranks=False,
    )
    expected = func[spin_summed, reordered, rank](vec, norb, nelec)
    np.testing.assert_allclose(rdm, expected, atol=1e-12)


def _rdm1_spin_summed(vec: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """Compute spin-summed 1-RDM directly from its definition."""
    rdm = np.zeros((norb, norb), dtype=complex)
    for i, j in itertools.combinations_with_replacement(range(norb), 2):
        op = ffsim.FermionOperator(
            {
                (ffsim.cre_a(i), ffsim.des_a(j)): 1,
                (ffsim.cre_b(i), ffsim.des_b(j)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[i, j] = val
        rdm[j, i] = val.conjugate()
    return rdm


def _rdm1(vec: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """Compute 1-RDM directly from its definition."""
    rdm = np.zeros((2 * norb, 2 * norb), dtype=complex)
    for i, j in itertools.combinations_with_replacement(range(norb), 2):
        op = ffsim.FermionOperator(
            {
                (ffsim.cre_a(i), ffsim.des_a(j)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[i, j] = val
        rdm[j, i] = val.conjugate()
        op = ffsim.FermionOperator(
            {
                (ffsim.cre_b(i), ffsim.des_b(j)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[norb + i, norb + j] = val
        rdm[norb + j, norb + i] = val.conjugate()
    return rdm


def _rdm2_spin_summed_reordered(
    vec: np.ndarray, norb: int, nelec: tuple[int, int]
) -> np.ndarray:
    """Compute spin-summed reordered 2-RDM directly from its definition."""
    rdm = np.zeros((norb, norb, norb, norb), dtype=complex)
    for p, q, r, s in itertools.product(range(norb), repeat=4):
        op = ffsim.FermionOperator(
            {
                (ffsim.cre_a(p), ffsim.cre_a(r), ffsim.des_a(s), ffsim.des_a(q)): 1,
                (ffsim.cre_a(p), ffsim.cre_b(r), ffsim.des_b(s), ffsim.des_a(q)): 1,
                (ffsim.cre_b(p), ffsim.cre_a(r), ffsim.des_a(s), ffsim.des_b(q)): 1,
                (ffsim.cre_b(p), ffsim.cre_b(r), ffsim.des_b(s), ffsim.des_b(q)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[p, q, r, s] = val
    return rdm


def _rdm2_spin_summed(vec: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """Compute spin-summed 2-RDM directly from its definition."""
    rdm = np.zeros((norb, norb, norb, norb), dtype=complex)
    for p, q, r, s in itertools.product(range(norb), repeat=4):
        op = ffsim.FermionOperator(
            {
                (ffsim.cre_a(p), ffsim.des_a(q), ffsim.cre_a(r), ffsim.des_a(s)): 1,
                (ffsim.cre_a(p), ffsim.des_a(q), ffsim.cre_b(r), ffsim.des_b(s)): 1,
                (ffsim.cre_b(p), ffsim.des_b(q), ffsim.cre_a(r), ffsim.des_a(s)): 1,
                (ffsim.cre_b(p), ffsim.des_b(q), ffsim.cre_b(r), ffsim.des_b(s)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[p, q, r, s] = val
    return rdm


def _rdm2_reordered(vec: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """Compute reordered 2-RDM directly from its definition."""
    rdm = np.zeros((2 * norb, 2 * norb, 2 * norb, 2 * norb), dtype=complex)
    for p, q, r, s in itertools.product(range(norb), repeat=4):
        op = ffsim.FermionOperator(
            {
                (ffsim.cre_a(p), ffsim.cre_a(r), ffsim.des_a(s), ffsim.des_a(q)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[p, q, r, s] = val

        op = ffsim.FermionOperator(
            {
                (ffsim.cre_a(p), ffsim.cre_b(r), ffsim.des_b(s), ffsim.des_a(q)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[p, q, norb + r, norb + s] = val

        op = ffsim.FermionOperator(
            {
                (ffsim.cre_b(p), ffsim.cre_a(r), ffsim.des_a(s), ffsim.des_b(q)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[norb + p, norb + q, r, s] = val

        op = ffsim.FermionOperator(
            {
                (ffsim.cre_b(p), ffsim.cre_b(r), ffsim.des_b(s), ffsim.des_b(q)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[norb + p, norb + q, norb + r, norb + s] = val
    return rdm


def _rdm2(vec: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """Compute 2-RDM directly from its definition."""
    rdm = np.zeros((2 * norb, 2 * norb, 2 * norb, 2 * norb), dtype=complex)
    for p, q, r, s in itertools.product(range(norb), repeat=4):
        op = ffsim.FermionOperator(
            {
                (ffsim.cre_a(p), ffsim.des_a(q), ffsim.cre_a(r), ffsim.des_a(s)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[p, q, r, s] = val

        op = ffsim.FermionOperator(
            {
                (ffsim.cre_a(p), ffsim.des_a(q), ffsim.cre_b(r), ffsim.des_b(s)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[p, q, norb + r, norb + s] = val

        op = ffsim.FermionOperator(
            {
                (ffsim.cre_b(p), ffsim.des_b(q), ffsim.cre_a(r), ffsim.des_a(s)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[norb + p, norb + q, r, s] = val

        op = ffsim.FermionOperator(
            {
                (ffsim.cre_b(p), ffsim.des_b(q), ffsim.cre_b(r), ffsim.des_b(s)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[norb + p, norb + q, norb + r, norb + s] = val
    return rdm
