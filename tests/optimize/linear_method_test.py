# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for linear method optimization algorithm."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pyscf
import pytest
from pyscf import cc

import ffsim


def test_minimize_linear_method():
    # Build an H2 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["H", (0, 0, 0)], ["H", (0, 0, 1.8)]],
        basis="sto-6g",
    )
    hartree_fock = pyscf.scf.RHF(mol)
    hartree_fock.kernel()

    # Get CCSD t2 amplitudes for initializing the ansatz
    ccsd = cc.CCSD(hartree_fock)
    _, _, t2 = ccsd.kernel()

    # Construct UCJ operator
    n_reps = 2
    operator = ffsim.UCJOperator.from_t_amplitudes(t2, n_reps=n_reps)
    params = operator.to_parameters()
    rng = np.random.default_rng(1234)
    x0 = rng.uniform(-10, 10, size=len(params))

    # Get molecular data and molecular Hamiltonian (one- and two-body tensors)
    mol_data = ffsim.MolecularData.from_scf(hartree_fock)
    norb = mol_data.norb
    nelec = mol_data.nelec
    mol_hamiltonian = mol_data.hamiltonian
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
    reference_state = ffsim.hartree_fock_state(norb, nelec)

    def params_to_vec(x: np.ndarray):
        operator = ffsim.UCJOperator.from_parameters(x, norb=norb, n_reps=n_reps)
        return ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

    def energy(x: np.ndarray):
        vec = params_to_vec(x)
        return np.real(np.vdot(vec, hamiltonian @ vec))

    info = defaultdict(list)

    def callback(intermediate_result):
        info["x"].append(intermediate_result.x)
        info["fun"].append(intermediate_result.fun)
        np.testing.assert_allclose(
            energy(intermediate_result.x), intermediate_result.fun
        )
        if hasattr(intermediate_result, "jac"):
            info["jac"].append(intermediate_result.jac)
        if hasattr(intermediate_result, "regularization"):
            info["regularization"].append(intermediate_result.regularization)
        if hasattr(intermediate_result, "variation"):
            info["variation"].append(intermediate_result.variation)

    # default optimization
    result = ffsim.optimize.minimize_linear_method(
        params_to_vec, x0=x0, hamiltonian=hamiltonian, callback=callback
    )
    np.testing.assert_allclose(energy(result.x), result.fun)
    np.testing.assert_allclose(result.fun, -0.970773)
    np.testing.assert_allclose(info["fun"][0], -0.834889, atol=1e-5)
    np.testing.assert_allclose(info["fun"][-1], -0.970773, atol=1e-5)
    for params, fun in zip(info["x"], info["fun"]):
        np.testing.assert_allclose(energy(params), fun)
    assert result.nit <= 7
    assert result.nit < result.nlinop < result.nfev

    # optimization with optimizing hyperparameters
    info = defaultdict(list)
    result = ffsim.optimize.minimize_linear_method(
        params_to_vec,
        x0=x0,
        hamiltonian=hamiltonian,
        regularization=0.01,
        variation=0.9,
        optimize_hyperparameters=False,
        callback=callback,
    )
    np.testing.assert_allclose(energy(result.x), result.fun)
    np.testing.assert_allclose(result.fun, -0.970773)
    for params, fun in zip(info["x"], info["fun"]):
        np.testing.assert_allclose(energy(params), fun)
    assert result.nit <= 11
    assert result.nit < result.nlinop < result.nfev
    assert set(info["regularization"]) == {0.01}
    assert set(info["variation"]) == {0.9}

    # optimization with maxiter
    info = defaultdict(list)
    result = ffsim.optimize.minimize_linear_method(
        params_to_vec, hamiltonian=hamiltonian, x0=x0, maxiter=3, callback=callback
    )
    assert result.nit == 3
    assert len(info["x"]) == 3
    assert len(info["fun"]) == 3
    assert len(info["jac"]) == 2
    np.testing.assert_allclose(energy(result.x), result.fun)

    # test raising errors
    with pytest.raises(ValueError, match="regularization"):
        result = ffsim.optimize.minimize_linear_method(
            params_to_vec, x0=x0, hamiltonian=hamiltonian, regularization=-1
        )
    with pytest.raises(ValueError, match="variation"):
        result = ffsim.optimize.minimize_linear_method(
            params_to_vec, x0=x0, hamiltonian=hamiltonian, variation=0
        )
    with pytest.raises(ValueError, match="variation"):
        result = ffsim.optimize.minimize_linear_method(
            params_to_vec, x0=x0, hamiltonian=hamiltonian, variation=1
        )
    with pytest.raises(ValueError, match="maxiter"):
        result = ffsim.optimize.minimize_linear_method(
            params_to_vec, x0=x0, hamiltonian=hamiltonian, maxiter=0
        )
