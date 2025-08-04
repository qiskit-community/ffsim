# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for linear algebra utilities."""

import numpy as np
import pytest

import ffsim
from ffsim.linalg.util import (
    antihermitian_from_parameters,
    antihermitian_from_parameters_jax,
    antihermitian_to_parameters,
    antihermitians_from_parameters,
    antihermitians_from_parameters_jax,
    antihermitians_to_parameters,
    df_tensors_from_params,
    df_tensors_from_params_jax,
    df_tensors_to_params,
    real_symmetric_from_parameters,
    real_symmetric_from_parameters_jax,
    real_symmetric_to_parameters,
    unitaries_from_parameters,
    unitaries_from_parameters_jax,
    unitaries_to_parameters,
    unitary_from_parameters,
    unitary_from_parameters_jax,
    unitary_to_parameters,
)

RNG = np.random.default_rng(203074816663819213199997065366693085494)


@pytest.mark.parametrize("dim", range(5))
@pytest.mark.parametrize("real", [True, False])
def test_antihermitian_parameters(dim: int, real: bool):
    """Test parameterizing antihermitian matrix."""
    mat = ffsim.random.random_antihermitian(dim, seed=RNG)
    if real:
        mat = mat.real
    params = antihermitian_to_parameters(mat, real=real)
    mat_roundtrip = antihermitian_from_parameters(params, dim, real=real)
    np.testing.assert_allclose(mat_roundtrip, mat)

    n_params = dim * (dim - 1) // 2 if real else dim**2
    params = RNG.normal(size=n_params)
    mat = antihermitian_from_parameters(params, dim, real=real)
    params_roundtrip = antihermitian_to_parameters(mat, real=real)
    np.testing.assert_allclose(params_roundtrip, params)
    assert ffsim.linalg.is_antihermitian(mat)


@pytest.mark.parametrize("dim", range(1, 5))
@pytest.mark.parametrize("real", [True, False])
def test_antihermitian_parameters_jax_consistent(dim: int, real: bool):
    """Test JAX and NumPy versions of parameterizing antihermitian give same results."""
    n_params = dim * (dim - 1) // 2 if real else dim**2
    params = RNG.normal(size=n_params)
    mat_numpy = antihermitian_from_parameters(params, dim, real=real)
    mat_jax = antihermitian_from_parameters_jax(params, dim, real=real)
    np.testing.assert_allclose(mat_jax, mat_numpy)


@pytest.mark.parametrize("dim", range(5))
@pytest.mark.parametrize("n_mats", range(1, 4))
@pytest.mark.parametrize("real", [True, False])
def test_antihermitians_parameters(dim: int, n_mats: int, real: bool):
    """Test parameterizing batch of antihermitian matrices."""
    mats = np.stack(
        [ffsim.random.random_antihermitian(dim, seed=RNG) for _ in range(n_mats)]
    )
    if real:
        mats = mats.real

    params = antihermitians_to_parameters(mats, real=real)
    mats_roundtrip = antihermitians_from_parameters(params, dim, n_mats, real=real)
    np.testing.assert_allclose(mats_roundtrip, mats)

    n_params_per_mat = dim * (dim - 1) // 2 if real else dim**2
    n_params_total = n_mats * n_params_per_mat
    params = RNG.normal(size=n_params_total)
    mats = antihermitians_from_parameters(params, dim, n_mats, real=real)
    params_roundtrip = antihermitians_to_parameters(mats, real=real)
    np.testing.assert_allclose(params_roundtrip, params)
    for i in range(n_mats):
        assert ffsim.linalg.is_antihermitian(mats[i])


@pytest.mark.parametrize("dim", range(5))
@pytest.mark.parametrize("n_mats", range(1, 4))
@pytest.mark.parametrize("real", [True, False])
def test_antihermitians_parameters_jax_consistent(dim: int, n_mats: int, real: bool):
    """Test JAX and NumPy versions of parameterizing antihermitian give same results."""
    n_params_per_mat = dim * (dim - 1) // 2 if real else dim**2
    params = RNG.normal(size=n_mats * n_params_per_mat)
    mat_numpy = antihermitians_from_parameters(params, dim, n_mats, real=real)
    mat_jax = antihermitians_from_parameters_jax(params, dim, n_mats, real=real)
    np.testing.assert_allclose(mat_jax, mat_numpy)


@pytest.mark.parametrize("dim", range(5))
@pytest.mark.parametrize("n_mats", range(1, 4))
@pytest.mark.parametrize("real", [True, False])
def test_antihermitians_consistent(dim: int, n_mats: int, real: bool):
    """Test that batch function gives same result as single matrix functions."""
    mats = np.array(
        [ffsim.random.random_antihermitian(dim, seed=RNG) for _ in range(n_mats)]
    )
    if real:
        mats = mats.real

    params_batch = antihermitians_to_parameters(mats, real=real)
    params_individual = np.concatenate(
        [antihermitian_to_parameters(mats[i], real=real) for i in range(n_mats)]
    )
    np.testing.assert_allclose(params_batch, params_individual)

    n_params_per_mat = dim * (dim - 1) // 2 if real else dim**2
    n_params_total = n_mats * n_params_per_mat
    params = RNG.normal(size=n_params_total)
    mats_batch = antihermitians_from_parameters(params, dim, n_mats, real=real)
    mats_individual = np.zeros_like(mats_batch)
    for i in range(n_mats):
        mats_individual[i] = antihermitian_from_parameters(
            params[i * n_params_per_mat : (i + 1) * n_params_per_mat], dim, real=real
        )
    np.testing.assert_allclose(mats_batch, mats_individual)


@pytest.mark.parametrize("dim", range(1, 5))
@pytest.mark.parametrize("real", [True, False])
def test_unitary_parameters(dim: int, real: bool):
    """Test parameterizing unitary matrix."""
    n_params = dim * (dim - 1) // 2 if real else dim**2

    if not real:
        # Unitary doesn't roundtrip for real orthogonal matrices because the
        # matrix logarithm has both real and imaginary parts
        mat = ffsim.random.random_unitary(dim, seed=RNG)
        params = unitary_to_parameters(mat, real=real)
        mat_roundtrip = unitary_from_parameters(params, dim, real=real)
        np.testing.assert_allclose(mat_roundtrip, mat)

    params = RNG.normal(size=n_params, scale=0.1)
    mat = unitary_from_parameters(params, dim, real=real)
    params_roundtrip = unitary_to_parameters(mat, real=real)
    np.testing.assert_allclose(params_roundtrip, params)
    assert ffsim.linalg.is_unitary(mat)


@pytest.mark.parametrize("dim", range(1, 5))
@pytest.mark.parametrize("real", [True, False])
def test_unitary_parameters_jax_consistent(dim: int, real: bool):
    """Test JAX and NumPy versions of parameterizing unitary give same results."""
    n_params = dim * (dim - 1) // 2 if real else dim**2
    params = RNG.normal(size=n_params)
    mat_numpy = unitary_from_parameters(params, dim, real=real)
    mat_jax = unitary_from_parameters_jax(params, dim, real=real)
    np.testing.assert_allclose(mat_jax, mat_numpy)


@pytest.mark.parametrize("dim", range(1, 5))
@pytest.mark.parametrize("n_mats", range(1, 4))
@pytest.mark.parametrize("real", [True, False])
def test_unitaries_parameters(dim: int, n_mats: int, real: bool):
    """Test parameterizing batch of unitary matrices."""
    mats = np.stack([ffsim.random.random_unitary(dim, seed=RNG) for _ in range(n_mats)])
    if real:
        mats = mats.real

    if not real:
        # Unitary doesn't roundtrip for real orthogonal matrices because the
        # matrix logarithm has both real and imaginary parts
        params = unitaries_to_parameters(mats, real=real)
        mats_roundtrip = unitaries_from_parameters(params, dim, n_mats, real=real)
        np.testing.assert_allclose(mats_roundtrip, mats)

    n_params_per_mat = dim * (dim - 1) // 2 if real else dim**2
    n_params_total = n_mats * n_params_per_mat
    params = RNG.normal(size=n_params_total, scale=0.1)
    mats = unitaries_from_parameters(params, dim, n_mats, real=real)
    params_roundtrip = unitaries_to_parameters(mats, real=real)
    np.testing.assert_allclose(params_roundtrip, params)
    for i in range(n_mats):
        assert ffsim.linalg.is_unitary(mats[i])


@pytest.mark.parametrize("dim", range(5))
@pytest.mark.parametrize("n_mats", range(1, 4))
@pytest.mark.parametrize("real", [True, False])
def test_unitaries_parameters_jax_consistent(dim: int, n_mats: int, real: bool):
    """Test JAX and NumPy versions of parameterizing antihermitian give same results."""
    n_params_per_mat = dim * (dim - 1) // 2 if real else dim**2
    params = RNG.normal(size=n_mats * n_params_per_mat, scale=0.1)
    mat_numpy = unitaries_from_parameters(params, dim, n_mats, real=real)
    mat_jax = unitaries_from_parameters_jax(params, dim, n_mats, real=real)
    np.testing.assert_allclose(mat_jax, mat_numpy)


@pytest.mark.parametrize("dim", range(1, 5))
@pytest.mark.parametrize("n_mats", range(1, 4))
@pytest.mark.parametrize("real", [True, False])
def test_unitaries_consistent(dim: int, n_mats: int, real: bool):
    """Test that batch function gives same result as single matrix functions."""
    mats = np.array([ffsim.random.random_unitary(dim, seed=RNG) for _ in range(n_mats)])
    if real:
        mats = mats.real

    params_batch = unitaries_to_parameters(mats, real=real)
    params_individual = np.concatenate(
        [unitary_to_parameters(mats[i], real=real) for i in range(n_mats)]
    )
    np.testing.assert_allclose(params_batch, params_individual)

    n_params_per_mat = dim * (dim - 1) // 2 if real else dim**2
    n_params_total = n_mats * n_params_per_mat
    params = RNG.normal(size=n_params_total, scale=0.1)
    mats_batch = unitaries_from_parameters(params, dim, n_mats, real=real)
    mats_individual = np.zeros_like(mats_batch)
    for i in range(n_mats):
        mats_individual[i] = unitary_from_parameters(
            params[i * n_params_per_mat : (i + 1) * n_params_per_mat], dim, real=real
        )
    np.testing.assert_allclose(mats_batch, mats_individual)


@pytest.mark.parametrize("dim", range(5))
def test_real_symmetric_parameters(dim: int):
    """Test parameterizing real symmetric matrix."""
    mat = ffsim.random.random_real_symmetric_matrix(dim, seed=RNG)
    params = real_symmetric_to_parameters(mat)
    mat_roundtrip = real_symmetric_from_parameters(params, dim)
    np.testing.assert_allclose(mat_roundtrip, mat)

    n_params = dim * (dim + 1) // 2
    params = RNG.normal(size=n_params)
    mat = real_symmetric_from_parameters(params, dim)
    params_roundtrip = real_symmetric_to_parameters(mat)
    np.testing.assert_allclose(params_roundtrip, params)
    assert ffsim.linalg.is_real_symmetric(mat)


@pytest.mark.parametrize("dim", range(1, 5))
def test_real_symmetric_parameters_custom_indices(dim: int):
    """Test parameterizing real symmetric matrix with custom indices."""
    triu_indices = [(p, p) for p in range(dim)]
    triu_indices.extend([(p, p + 1) for p in range(dim - 1)])
    rows, cols = zip(*triu_indices)
    triu_mask = np.zeros((dim, dim), dtype=bool)
    triu_mask[rows, cols] = True
    triu_mask[cols, rows] = True

    mat = ffsim.random.random_real_symmetric_matrix(dim, seed=RNG)
    params = real_symmetric_to_parameters(mat, triu_indices)
    mat_roundtrip = real_symmetric_from_parameters(params, dim, triu_indices)
    np.testing.assert_allclose(mat_roundtrip, mat * triu_mask)

    n_params = len(triu_indices)
    params = RNG.normal(size=n_params)
    mat = real_symmetric_from_parameters(params, dim, triu_indices)
    params_roundtrip = real_symmetric_to_parameters(mat, triu_indices)
    np.testing.assert_allclose(params_roundtrip, params)
    assert ffsim.linalg.is_real_symmetric(mat)


@pytest.mark.parametrize("dim", range(5))
def test_real_symmetric_parameters_jax_consistent(dim: int):
    """Test JAX and NumPy versions of parameterizing symmetric mat give same results."""
    n_params = dim * (dim + 1) // 2
    params = RNG.normal(size=n_params)
    mat_numpy = real_symmetric_from_parameters(params, dim)
    mat_jax = real_symmetric_from_parameters_jax(params, dim)
    np.testing.assert_allclose(mat_jax, mat_numpy)


@pytest.mark.parametrize("dim", range(1, 5))
def test_real_symmetric_parameters_custom_indices_jax_consistent(dim: int):
    """Test JAX and NumPy versions of symmetric mat with indices give same results."""
    triu_indices = [(p, p) for p in range(dim)]
    triu_indices.extend([(p, p + 1) for p in range(dim - 1)])
    n_params = len(triu_indices)
    params = RNG.normal(size=n_params)
    mat_numpy = real_symmetric_from_parameters(params, dim, triu_indices)
    mat_jax = real_symmetric_from_parameters_jax(params, dim, triu_indices)
    np.testing.assert_allclose(mat_jax, mat_numpy)


@pytest.mark.parametrize("n_tensors", range(1, 4))
@pytest.mark.parametrize("norb", range(1, 5))
@pytest.mark.parametrize("real", [True, False])
def test_df_tensors_parameters(n_tensors: int, norb: int, real: bool):
    """Test parameterizing double-factorization tensors."""
    n_params_per_orb_rot = norb * (norb - 1) // 2 if real else norb**2
    n_params_per_diag_coulomb = norb * (norb + 1) // 2
    n_params_total = n_tensors * (n_params_per_orb_rot + n_params_per_diag_coulomb)
    params = RNG.normal(size=n_params_total, scale=0.1)
    diag_coulomb_mats, orbital_rotations = df_tensors_from_params(
        params, n_tensors, norb, real=real
    )
    params_roundtrip = df_tensors_to_params(
        diag_coulomb_mats, orbital_rotations, real=real
    )
    np.testing.assert_allclose(params_roundtrip, params)


@pytest.mark.parametrize("n_tensors", range(1, 4))
@pytest.mark.parametrize("norb", range(1, 5))
@pytest.mark.parametrize("real", [True, False])
def test_df_tensors_parameters_custom_indices(n_tensors: int, norb: int, real: bool):
    """Test parameterizing double-factorization tensors with custom indices."""
    diag_coulomb_indices = [(p, p) for p in range(norb)]
    diag_coulomb_indices.extend([(p, p + 1) for p in range(norb - 1)])

    n_params_per_orb_rot = norb * (norb - 1) // 2 if real else norb**2
    n_params_per_diag_coulomb = len(diag_coulomb_indices)
    n_params_total = n_tensors * (n_params_per_orb_rot + n_params_per_diag_coulomb)
    params = RNG.normal(size=n_params_total, scale=0.1)
    diag_coulomb_mats, orbital_rotations = df_tensors_from_params(
        params, n_tensors, norb, diag_coulomb_indices, real=real
    )
    params_roundtrip = df_tensors_to_params(
        diag_coulomb_mats, orbital_rotations, diag_coulomb_indices, real=real
    )
    np.testing.assert_allclose(params, params_roundtrip)


@pytest.mark.parametrize("n_tensors", range(1, 4))
@pytest.mark.parametrize("norb", range(1, 5))
@pytest.mark.parametrize("real", [True, False])
def test_df_tensors_parameters_jax_consistency(n_tensors: int, norb: int, real: bool):
    """Test JAX and NumPy versions of parameterizing DF tensors give same results."""
    n_params_per_orb_rot = norb * (norb - 1) // 2 if real else norb**2
    n_params_per_diag_coulomb = norb * (norb + 1) // 2
    n_params_total = n_tensors * (n_params_per_orb_rot + n_params_per_diag_coulomb)
    params = RNG.normal(size=n_params_total)
    diag_coulomb_numpy, orb_rot_numpy = df_tensors_from_params(
        params, n_tensors, norb, real=real
    )
    diag_coulomb_jax, orb_rot_jax = df_tensors_from_params_jax(
        params, n_tensors, norb, real=real
    )
    np.testing.assert_allclose(diag_coulomb_jax, diag_coulomb_numpy)
    np.testing.assert_allclose(orb_rot_jax, orb_rot_numpy)


@pytest.mark.parametrize("n_tensors", range(1, 4))
@pytest.mark.parametrize("norb", range(1, 5))
@pytest.mark.parametrize("real", [True, False])
def test_df_tensors_parameters_custom_indices_jax_consistency(
    n_tensors: int, norb: int, real: bool
):
    """Test JAX and NumPy versions of DF tensors with indices give same results."""
    diag_coulomb_indices = [(p, p) for p in range(norb)]
    diag_coulomb_indices.extend([(p, p + 1) for p in range(norb - 1)])

    n_params_per_orb_rot = norb * (norb - 1) // 2 if real else norb**2
    n_params_per_diag_coulomb = len(diag_coulomb_indices)
    n_params_total = n_tensors * (n_params_per_orb_rot + n_params_per_diag_coulomb)
    params = RNG.normal(size=n_params_total)
    diag_coulomb_numpy, orb_rot_numpy = df_tensors_from_params(
        params, n_tensors, norb, diag_coulomb_indices, real=real
    )
    diag_coulomb_jax, orb_rot_jax = df_tensors_from_params_jax(
        params, n_tensors, norb, diag_coulomb_indices, real=real
    )
    np.testing.assert_allclose(diag_coulomb_jax, diag_coulomb_numpy)
    np.testing.assert_allclose(orb_rot_jax, orb_rot_numpy)
