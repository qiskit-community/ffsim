# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for double-factorized Hamiltonian."""

from __future__ import annotations

import itertools
from unittest.mock import Mock, patch

import numpy as np
import pytest
import scipy.optimize

import ffsim

RNG = np.random.default_rng(73020557092643299885827653110468818597)


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("method", [None, "SLSQP"])
def test_from_molecular_hamiltonian_optimize_kwargs(optimize: bool, method: str | None):
    hamiltonian = ffsim.random.random_molecular_hamiltonian(2, seed=128, dtype=float)
    callback = Mock()
    bounds = scipy.optimize.Bounds(-2, 2)
    constraint = scipy.optimize.NonlinearConstraint(lambda x: np.sum(x**2), 0, 100)
    options = dict(maxiter=2)
    kwargs: dict = dict(
        callback=callback, bounds=bounds, constraints=constraint, options=options
    )
    if method is not None:
        kwargs["method"] = method

    # Observe every SciPy invocation without solving each intermediate optimization.
    with patch(
        "scipy.optimize.minimize",
        side_effect=lambda fun, x0, **kwargs: scipy.optimize.OptimizeResult(x=x0),
    ) as minimize:
        ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
            hamiltonian, max_vecs=1, optimize=optimize, **kwargs
        )

    assert minimize.call_count == (1 if optimize else 0)
    for call in minimize.call_args_list:
        assert call.kwargs["method"] == (method or "L-BFGS-B")
        assert call.kwargs["callback"] is callback
        assert call.kwargs["bounds"] is bounds
        assert call.kwargs["constraints"] is constraint
        assert call.kwargs["options"] is options
    assert options == {"maxiter": 2}
    assert ("method" in kwargs) == (method is not None)


def test_from_molecular_hamiltonian_optimize_no_terms():
    """Test optimize with a two-body tensor that has no terms."""
    norb = 3
    hamiltonian = ffsim.MolecularHamiltonian(
        one_body_tensor=ffsim.random.random_hermitian(norb, seed=RNG).real,
        two_body_tensor=np.zeros((norb, norb, norb, norb)),
    )
    df_hamiltonian = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
        hamiltonian, optimize=True
    )
    assert df_hamiltonian.diag_coulomb_mats.shape == (0, norb, norb)
    assert df_hamiltonian.orbital_rotations.shape == (0, norb, norb)
    np.testing.assert_allclose(
        df_hamiltonian.one_body_tensor, hamiltonian.one_body_tensor
    )


@pytest.mark.parametrize("z_representation", [False, True])
def test_linear_operator(z_representation: bool):
    """Test linear operator."""
    norb = 4
    nelec = (2, 2)

    dim = ffsim.dim(norb, nelec)
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(
        norb, seed=RNG, dtype=float
    )

    df_hamiltonian = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_hamiltonian,
        z_representation=z_representation,
    )

    actual_linop = ffsim.linear_operator(df_hamiltonian, norb, nelec)
    expected_linop = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    dim = ffsim.dim(norb, nelec)

    state = ffsim.random.random_state_vector(dim, seed=RNG)
    actual = actual_linop @ state
    expected = expected_linop @ state
    np.testing.assert_allclose(actual, expected)

    state = ffsim.random.random_state_vector(dim, seed=RNG, dtype=float)
    actual = actual_linop @ state
    expected = expected_linop @ state
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("norb", range(5))
def test_z_representation_round_trip(norb: int):
    """Test converting to and from Z-representation"""
    mol_ham = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG, dtype=float)
    df_hamiltonian = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_ham
    )
    df_hamiltonian_roundtrip = (
        df_hamiltonian.to_z_representation().to_number_representation()
    )
    assert ffsim.approx_eq(df_hamiltonian_roundtrip, df_hamiltonian)


@pytest.mark.parametrize("z_representation", [False, True])
def test_to_molecular_hamiltonian_roundtrip(z_representation: bool):
    """Test converting to molecular Hamiltonian."""
    norb = 5
    hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG, dtype=float)
    hamiltonian_roundtrip = (
        ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
            hamiltonian, z_representation=z_representation
        ).to_molecular_hamiltonian()
    )
    assert ffsim.approx_eq(hamiltonian_roundtrip, hamiltonian)


@pytest.mark.parametrize("z_representation", [False, True])
def test_to_molecular_hamiltonian_symmetry(z_representation: bool):
    """Test molecular Hamiltonian symmetry."""
    norb = 5
    df_hamiltonian = ffsim.random.random_double_factorized_hamiltonian(
        norb, z_representation=z_representation, seed=RNG
    )
    mol_hamiltonian = df_hamiltonian.to_molecular_hamiltonian()
    two_body_tensor = mol_hamiltonian.two_body_tensor
    for i, j, k, ell in itertools.product(range(norb), repeat=4):
        val = two_body_tensor[i, j, k, ell]
        np.testing.assert_allclose(two_body_tensor[k, ell, i, j], val)
        np.testing.assert_allclose(two_body_tensor[j, i, ell, k], val.conjugate())
        np.testing.assert_allclose(two_body_tensor[ell, k, j, i], val.conjugate())


def test_diag():
    """Test computing diagonal."""
    norb = 5
    nelec = (3, 2)
    # TODO remove dtype=float once complex is supported
    hamiltonian = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
        ffsim.random.random_molecular_hamiltonian(norb, seed=RNG, dtype=float)
    )
    linop = ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)
    hamiltonian_dense = linop @ np.eye(ffsim.dim(norb, nelec))
    np.testing.assert_allclose(
        ffsim.diag(hamiltonian, norb=norb, nelec=nelec), np.diag(hamiltonian_dense)
    )


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (4, (2, 2)),
        (4, (1, 2)),
        (4, (0, 2)),
        (4, (0, 0)),
    ],
)
@pytest.mark.parametrize("real", [True, False])
def test_fermion_operator(norb: int, nelec: tuple[int, int], real: bool):
    """Test FermionOperator."""
    df_hamiltonian = ffsim.random.random_double_factorized_hamiltonian(
        norb, real=real, seed=RNG
    )
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)

    op = ffsim.fermion_operator(df_hamiltonian)
    linop = ffsim.linear_operator(op, norb, nelec)
    expected_linop = ffsim.linear_operator(df_hamiltonian, norb, nelec)

    actual = linop @ vec
    expected = expected_linop @ vec
    np.testing.assert_allclose(actual, expected)
