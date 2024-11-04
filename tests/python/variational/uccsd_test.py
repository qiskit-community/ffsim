# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for UCCSD ansatz."""

import dataclasses

import numpy as np

import ffsim


def test_norb():
    """Test norb property."""
    rng = np.random.default_rng(4878)
    norb = 5
    nocc = 3
    operator = ffsim.random.random_uccsd_restricted(norb, nocc, real=True, seed=rng)
    assert operator.norb == norb


def test_n_params():
    """Test computing number of parameters."""
    rng = np.random.default_rng(4878)
    norb = 5
    nocc = 3
    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_uccsd_restricted(
            norb,
            nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
            real=True,
            seed=rng,
        )
        actual = ffsim.UCCSDOpRestrictedReal.n_params(
            norb, nocc, with_final_orbital_rotation=with_final_orbital_rotation
        )
        expected = len(operator.to_parameters())
        assert actual == expected


def test_parameters_roundtrip():
    """Test parameters roundtrip."""
    rng = np.random.default_rng(4623)
    norb = 5
    nocc = 3
    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_uccsd_restricted(
            norb,
            nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
            real=True,
            seed=rng,
        )
        roundtripped = ffsim.UCCSDOpRestrictedReal.from_parameters(
            operator.to_parameters(),
            norb=norb,
            nocc=nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        assert ffsim.approx_eq(roundtripped, operator)


def test_approx_eq():
    """Test approximate equality."""
    rng = np.random.default_rng(4623)
    norb = 5
    nocc = 3
    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_uccsd_restricted(
            norb,
            nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
            real=True,
            seed=rng,
        )
        roundtripped = ffsim.UCCSDOpRestrictedReal.from_parameters(
            operator.to_parameters(),
            norb=norb,
            nocc=nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        assert ffsim.approx_eq(operator, roundtripped)
        assert not ffsim.approx_eq(
            operator, dataclasses.replace(operator, t1=2 * operator.t1)
        )
        assert not ffsim.approx_eq(
            operator, dataclasses.replace(operator, t2=2 * operator.t2)
        )


def test_apply_unitary():
    """Test unitary."""
    rng = np.random.default_rng(4623)
    norb = 5
    nocc = 3
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, (nocc, nocc)), seed=rng)
    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_uccsd_restricted(
            norb,
            nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
            real=True,
            seed=rng,
        )
        result = ffsim.apply_unitary(vec, operator, norb=norb, nelec=(nocc, nocc))
        np.testing.assert_allclose(np.linalg.norm(result), 1.0)
