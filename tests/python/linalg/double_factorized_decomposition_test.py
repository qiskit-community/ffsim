# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for double factorization utilities."""

from __future__ import annotations

import itertools
import warnings
from unittest.mock import Mock, patch

import numpy as np
import pyscf
import pyscf.cc
import pyscf.mcscf
import pytest
import scipy.optimize
from opt_einsum import contract

import ffsim
from ffsim.linalg.double_factorized_decomposition import (
    _double_factorized_t2_alpha_beta_explicit,
    _double_factorized_t2_explicit,
    optimal_diag_coulomb_mats,
    reconstruct_t2,
    reconstruct_t2_alpha_beta,
)

RNG = np.random.default_rng(139632037091916421993148931543991464292)


@pytest.mark.parametrize("dim", range(6))
def test_modified_cholesky(dim: int):
    """Test modified Cholesky decomposition on a random tensor."""
    # construct a random positive definite matrix
    unitary = np.array(ffsim.random.random_unitary(dim, seed=RNG))
    eigs = RNG.uniform(size=dim)
    mat = unitary @ np.diag(eigs) @ unitary.T.conj()
    cholesky_vecs = ffsim.linalg.modified_cholesky(mat)
    reconstructed = contract("ji,ki->jk", cholesky_vecs, cholesky_vecs.conj())
    np.testing.assert_allclose(reconstructed, mat, atol=1e-8)


@pytest.mark.parametrize(
    "dim, cholesky", list(itertools.product(range(6), [False, True]))
)
def test_double_factorized_random(dim: int, cholesky: bool):
    """Test double-factorized decomposition on a random tensor."""
    two_body_tensor = ffsim.random.random_two_body_tensor(dim, seed=RNG, dtype=float)
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized(
        two_body_tensor, cholesky=cholesky
    )
    reconstructed = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-8)


@pytest.mark.parametrize("cholesky", [True, False])
def test_double_factorized_tol_max_vecs(cholesky: bool):
    """Test double-factorized decomposition error threshold and max vecs."""
    mol = pyscf.gto.Mole()
    mol.build(
        verbose=0,
        atom=[["Li", (0, 0, 0)], ["H", (1.6, 0, 0)]],
        basis="sto-3g",
    )
    hartree_fock = pyscf.scf.RHF(mol)
    hartree_fock.kernel()
    norb = hartree_fock.mol.nao_nr()
    mc = pyscf.mcscf.CASCI(hartree_fock, norb, mol.nelec)
    two_body_tensor = pyscf.ao2mo.restore(1, mc.get_h2cas(), mc.ncas)

    # test max_vecs
    max_vecs = 20
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized(
        two_body_tensor, max_vecs=max_vecs, cholesky=cholesky
    )
    reconstructed = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    assert len(orbital_rotations) == max_vecs
    np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-5)

    # test error threshold
    tol = 1e-4
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized(
        two_body_tensor, tol=tol
    )
    reconstructed = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    assert len(orbital_rotations) <= 18
    np.testing.assert_allclose(reconstructed, two_body_tensor, atol=tol)

    # test error threshold and max vecs
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized(
        two_body_tensor, tol=tol, max_vecs=max_vecs
    )
    reconstructed = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    assert len(orbital_rotations) <= 18
    np.testing.assert_allclose(reconstructed, two_body_tensor, atol=tol)


def test_optimal_diag_coulomb_mats_exact():
    """Test optimal diag Coulomb matrices on exact decomposition."""
    dim = 5
    two_body_tensor = ffsim.random.random_two_body_tensor(dim, seed=RNG, dtype=float)

    _, orbital_rotations = ffsim.linalg.double_factorized(two_body_tensor)
    diag_coulomb_mats_optimal = optimal_diag_coulomb_mats(
        two_body_tensor, orbital_rotations
    )
    reconstructed = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats_optimal,
        orbital_rotations,
        orbital_rotations,
    )
    np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-8)


def test_optimal_diag_coulomb_mats_approximate():
    """Test optimal diag Coulomb matrices on approximate decomposition."""
    dim = 5
    two_body_tensor = ffsim.random.random_two_body_tensor(dim, seed=RNG, dtype=float)

    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized(
        two_body_tensor, max_vecs=3
    )
    diag_coulomb_mats_optimal = optimal_diag_coulomb_mats(
        two_body_tensor, orbital_rotations
    )
    reconstructed = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    reconstructed_optimal = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats_optimal,
        orbital_rotations,
        orbital_rotations,
    )
    error = np.sum((reconstructed - two_body_tensor) ** 2)
    error_optimal = np.sum((reconstructed_optimal - two_body_tensor) ** 2)
    assert error_optimal < error


@pytest.mark.parametrize(
    "compress, factorize",
    [
        (ffsim.linalg.double_factorized_compressed, ffsim.linalg.double_factorized),
        (
            ffsim.linalg.double_factorized_t2_compressed,
            ffsim.linalg.double_factorized_t2,
        ),
        (
            ffsim.linalg.double_factorized_t2_alpha_beta_compressed,
            ffsim.linalg.double_factorized_t2_alpha_beta,
        ),
    ],
    ids=["two_body", "t2", "t2_alpha_beta"],
)
@pytest.mark.parametrize("method", [None, "SLSQP"])
def test_double_factorized_compressed_optimize_kwargs(
    compress, factorize, method: str | None
):
    """Test that keyword arguments are passed through to scipy.optimize.minimize."""
    if compress is ffsim.linalg.double_factorized_compressed:
        tensor = ffsim.random.random_molecular_hamiltonian(
            2, seed=128, dtype=float
        ).two_body_tensor
        compress_kwargs: dict = dict(max_vecs=1)
        n_stages = 1
    else:
        tensor = np.ones((1, 1, 1, 1))
        compress_kwargs = dict(max_terms=1, multi_stage_start=2, multi_stage_step=1)
        n_stages = 2
    initial_guess = factorize(tensor)
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
        compress(tensor, initial_guess=initial_guess, **compress_kwargs, **kwargs)

    assert minimize.call_count == n_stages
    for call in minimize.call_args_list:
        assert call.kwargs["method"] == (method or "L-BFGS-B")
        assert call.kwargs["callback"] is callback
        assert call.kwargs["bounds"] is bounds
        assert call.kwargs["constraints"] is constraint
        assert call.kwargs["options"] is options
    assert options == {"maxiter": 2}
    assert ("method" in kwargs) == (method is not None)


@pytest.mark.parametrize(
    "compress, factorize, tensor, kwargs",
    [
        (
            ffsim.linalg.double_factorized_compressed,
            ffsim.linalg.double_factorized,
            ffsim.random.random_two_body_tensor(3, seed=2345, dtype=float),
            dict(max_vecs=2),
        ),
        (
            ffsim.linalg.double_factorized_t2_compressed,
            ffsim.linalg.double_factorized_t2,
            ffsim.random.random_t2_amplitudes(4, 2, seed=2345, dtype=float),
            dict(max_terms=2),
        ),
        (
            ffsim.linalg.double_factorized_t2_alpha_beta_compressed,
            ffsim.linalg.double_factorized_t2_alpha_beta,
            np.random.default_rng(2345).standard_normal((2, 1, 2, 3)),
            dict(max_terms=2),
        ),
    ],
    ids=["two_body", "t2", "t2_alpha_beta"],
)
def test_double_factorized_compressed_default_initial_guess(
    compress, factorize, tensor, kwargs
):
    """Test that the default initial guess is the exact factorization."""
    mats, rots, result = compress(tensor, **kwargs, options=dict(maxiter=5))
    expected_mats, expected_rots, expected_result = compress(
        tensor, initial_guess=factorize(tensor), **kwargs, options=dict(maxiter=5)
    )
    np.testing.assert_array_equal(mats, expected_mats)
    np.testing.assert_array_equal(rots, expected_rots)
    np.testing.assert_array_equal(result.x, expected_result.x)


def test_double_factorized_compressed_does_not_modify_initial_guess():
    """Test that the compression does not modify the initial guess in place."""
    two_body_tensor = ffsim.random.random_two_body_tensor(4, seed=1, dtype=float)
    mats, rots = ffsim.linalg.double_factorized(two_body_tensor, cholesky=False)
    # The test is only meaningful if some orbital rotations have determinant -1
    assert np.any(np.linalg.det(rots) < 0)
    mats_copy, rots_copy = mats.copy(), rots.copy()
    ffsim.linalg.double_factorized_compressed(
        two_body_tensor, initial_guess=(mats, rots), options=dict(maxiter=1)
    )
    np.testing.assert_array_equal(mats, mats_copy)
    np.testing.assert_array_equal(rots, rots_copy)


def test_double_factorized_compressed_validation():
    """Test input validation for the two-body tensor compression function."""
    two_body_tensor = ffsim.random.random_two_body_tensor(3, seed=RNG, dtype=float)
    mats, rots = ffsim.linalg.double_factorized(two_body_tensor)
    compress = ffsim.linalg.double_factorized_compressed
    with pytest.raises(ValueError, match="complex"):
        compress(two_body_tensor.astype(complex), initial_guess=(mats, rots))
    with pytest.raises(ValueError, match="max_vecs"):
        compress(two_body_tensor, initial_guess=(mats, rots), max_vecs=0)
    with pytest.raises(ValueError, match="at least one term"):
        compress(two_body_tensor, initial_guess=(mats[:0], rots[:0]))
    with pytest.raises(ValueError, match="at least one term"):
        compress(np.zeros((3, 3, 3, 3)))
    with pytest.raises(ValueError, match="shape"):
        compress(two_body_tensor, initial_guess=(mats, rots[:-1]))
    with pytest.raises(ValueError, match="shape"):
        compress(two_body_tensor, initial_guess=(mats[:, :2], rots))
    with pytest.raises(ValueError, match="lower triangular"):
        compress(
            two_body_tensor, initial_guess=(mats, rots), diag_coulomb_indices=[(1, 0)]
        )
    with pytest.raises(ValueError, match="real-valued"):
        compress(two_body_tensor, initial_guess=(mats, rots.astype(complex)))


def test_double_factorized_t2_compressed_validation():
    """Test input validation for the t2 compression function."""
    t2 = ffsim.random.random_t2_amplitudes(4, 2, seed=RNG, dtype=float)
    mats, rots = ffsim.linalg.double_factorized_t2(t2)
    compress = ffsim.linalg.double_factorized_t2_compressed
    with pytest.raises(ValueError, match="complex t2 amplitudes"):
        compress(t2.astype(complex), initial_guess=(mats, rots))
    with pytest.raises(ValueError, match="max_terms"):
        compress(t2, initial_guess=(mats, rots), max_terms=0)
    with pytest.raises(ValueError, match="at least one term"):
        compress(t2, initial_guess=(mats[:0], rots[:0]))
    with pytest.raises(ValueError, match="at least one term"):
        compress(np.zeros((2, 2, 2, 2)))
    with pytest.raises(ValueError, match="shape"):
        compress(t2, initial_guess=(mats, rots[:-1]))
    with pytest.raises(ValueError, match="shape"):
        compress(t2, initial_guess=(mats[:, :3, :3], rots))
    with pytest.raises(ValueError, match="lower triangular"):
        compress(t2, initial_guess=(mats, rots), diag_coulomb_indices=[(1, 0)])
    # Index validation happens before computing the default initial guess
    with patch(
        "ffsim.linalg.double_factorized_decomposition._double_factorized_t2_explicit"
    ) as factorize:
        with pytest.raises(ValueError, match="lower triangular"):
            compress(t2, diag_coulomb_indices=[(1, 0)])
    factorize.assert_not_called()


def test_double_factorized_t2_alpha_beta_compressed_validation():
    """Test input validation for the alpha-beta t2 compression function."""
    t2ab = RNG.standard_normal((2, 1, 2, 3))
    mats, rots = ffsim.linalg.double_factorized_t2_alpha_beta(t2ab)
    compress = ffsim.linalg.double_factorized_t2_alpha_beta_compressed
    with pytest.raises(ValueError, match="complex t2 amplitudes"):
        compress(t2ab.astype(complex), initial_guess=(mats, rots))
    with pytest.raises(ValueError, match="max_terms"):
        compress(t2ab, initial_guess=(mats, rots), max_terms=0)
    with pytest.raises(ValueError, match="at least one term"):
        compress(t2ab, initial_guess=(mats[:0], rots[:0]))
    with pytest.raises(ValueError, match="at least one term"):
        compress(np.zeros((2, 1, 2, 3)))
    with pytest.raises(ValueError, match="shape"):
        compress(t2ab, initial_guess=(mats, rots[:-1]))
    with pytest.raises(ValueError, match="shape"):
        compress(t2ab, initial_guess=(mats[:, :2], rots))
    for diag_coulomb_indices in [([(1, 0)], None, None), (None, None, [(1, 0)])]:
        with pytest.raises(ValueError, match="lower triangular"):
            compress(
                t2ab,
                initial_guess=(mats, rots),
                diag_coulomb_indices=diag_coulomb_indices,
            )
    # Alpha-beta indices may be lower triangular
    compress(
        t2ab,
        initial_guess=(mats, rots),
        diag_coulomb_indices=(None, [(1, 0)], None),
        options=dict(maxiter=1),
    )


def test_double_factorized_optimize_deprecated():
    """Test the deprecated optimize argument of double_factorized."""
    two_body_tensor = ffsim.random.random_two_body_tensor(3, seed=RNG, dtype=float)
    diag_coulomb_indices = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
    with pytest.deprecated_call(match="double_factorized_compressed"):
        mats, rots, result = ffsim.linalg.double_factorized(
            two_body_tensor,
            max_vecs=2,
            optimize=True,
            diag_coulomb_indices=diag_coulomb_indices,
            return_optimize_result=True,
            options=dict(maxiter=5),
        )
    expected_mats, expected_rots, expected_result = (
        ffsim.linalg.double_factorized_compressed(
            two_body_tensor,
            max_vecs=2,
            diag_coulomb_indices=diag_coulomb_indices,
            options=dict(maxiter=5),
        )
    )
    np.testing.assert_allclose(mats, expected_mats)
    np.testing.assert_allclose(rots, expected_rots)
    assert result is not None
    np.testing.assert_allclose(result.x, expected_result.x)


@pytest.mark.parametrize(
    "factorize, compress, t2, diag_coulomb_indices",
    [
        (
            ffsim.linalg.double_factorized_t2,
            ffsim.linalg.double_factorized_t2_compressed,
            ffsim.random.random_t2_amplitudes(4, 2, seed=5678, dtype=float),
            [(p, p + 1) for p in range(3)],
        ),
        (
            ffsim.linalg.double_factorized_t2_alpha_beta,
            ffsim.linalg.double_factorized_t2_alpha_beta_compressed,
            np.random.default_rng(5678).standard_normal((2, 1, 2, 3)),
            ([(0, 1)], [(0, 0), (1, 0)], None),
        ),
    ],
    ids=["t2", "t2_alpha_beta"],
)
def test_double_factorized_t2_optimize_deprecated(
    factorize, compress, t2, diag_coulomb_indices
):
    """Test the deprecated optimize argument of the t2 factorizations."""
    kwargs = dict(
        max_terms=2,
        diag_coulomb_indices=diag_coulomb_indices,
        regularization=1e-3,
        multi_stage_start=4,
        options=dict(maxiter=5),
    )
    with pytest.deprecated_call(match=compress.__name__):
        mats, rots, result = factorize(
            t2, optimize=True, return_optimize_result=True, **kwargs
        )
    expected_mats, expected_rots, expected_result = compress(t2, **kwargs)
    np.testing.assert_allclose(mats, expected_mats)
    np.testing.assert_allclose(rots, expected_rots)
    assert result is not None
    np.testing.assert_allclose(result.x, expected_result.x)


@pytest.mark.parametrize(
    "factorize, tensor, kwargs, names",
    [
        (
            ffsim.linalg.double_factorized,
            np.zeros((2, 2, 2, 2)),
            dict(optimize=False, return_optimize_result=True),
            ["return_optimize_result"],
        ),
        (
            ffsim.linalg.double_factorized_t2,
            np.zeros((1, 1, 1, 1)),
            dict(diag_coulomb_indices=[(0, 0)], options=dict(maxiter=1)),
            ["diag_coulomb_indices", "optimize_kwargs"],
        ),
        (
            ffsim.linalg.double_factorized_t2_alpha_beta,
            np.zeros((1, 1, 1, 1)),
            dict(regularization=1.0, multi_stage_start=2, multi_stage_step=1),
            ["regularization", "multi_stage_start", "multi_stage_step"],
        ),
    ],
    ids=["two_body", "t2", "t2_alpha_beta"],
)
def test_double_factorized_deprecation_message(factorize, tensor, kwargs, names):
    """Test that the deprecation warning names exactly the arguments that were set."""
    with pytest.deprecated_call() as record:
        factorize(tensor, **kwargs)
    (warning,) = record
    message = str(warning.message)
    for name in names:
        assert f"`{name}`" in message
    assert message.count("`") == 2 * (len(names) + 3)
    assert ("arguments" in message) == (len(names) > 1)


@pytest.mark.parametrize(
    "factorize, tensor",
    [
        (ffsim.linalg.double_factorized, np.ones((2, 2, 2, 2))),
        (ffsim.linalg.double_factorized_t2, np.ones((1, 1, 1, 1))),
        (ffsim.linalg.double_factorized_t2_alpha_beta, np.ones((1, 1, 1, 1))),
    ],
    ids=["two_body", "t2", "t2_alpha_beta"],
)
def test_double_factorized_no_deprecation_warning(factorize, tensor):
    """Test that no deprecation warning is raised without the deprecated arguments."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        factorize(tensor)


@pytest.mark.parametrize(
    "factorize, compress, tensor, factorize_kwargs, compress_kwargs",
    [
        (
            ffsim.linalg.double_factorized,
            ffsim.linalg.double_factorized_compressed,
            ffsim.random.random_two_body_tensor(3, seed=4567, dtype=float),
            dict(max_vecs=2),
            dict(max_vecs=2, diag_coulomb_indices=[(0, 0), (0, 1)]),
        ),
        (
            ffsim.linalg.double_factorized_t2,
            ffsim.linalg.double_factorized_t2_compressed,
            ffsim.random.random_t2_amplitudes(4, 2, seed=4567, dtype=float),
            {},
            dict(
                max_terms=2,
                diag_coulomb_indices=[(0, 0), (0, 1)],
                multi_stage_start=4,
                multi_stage_step=2,
                regularization=0.1,
            ),
        ),
        (
            ffsim.linalg.double_factorized_t2_alpha_beta,
            ffsim.linalg.double_factorized_t2_alpha_beta_compressed,
            np.random.default_rng(4567).standard_normal((2, 1, 2, 3)),
            {},
            dict(
                max_terms=2,
                diag_coulomb_indices=([(0, 1)], [(2, 1)], [(0, 0), (1, 2)]),
                multi_stage_start=4,
                multi_stage_step=2,
                regularization=0.1,
            ),
        ),
    ],
    ids=["two_body", "t2", "t2_alpha_beta"],
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_double_factorized_optimize_matches_compressed(
    factorize, compress, tensor, factorize_kwargs, compress_kwargs
):
    """Test that the deprecated optimize argument matches the compression function."""
    mats, rots, result = factorize(
        tensor,
        optimize=True,
        return_optimize_result=True,
        options=dict(maxiter=5),
        **compress_kwargs,
    )
    expected_mats, expected_rots, expected_result = compress(
        tensor,
        initial_guess=factorize(tensor, **factorize_kwargs),
        options=dict(maxiter=5),
        **compress_kwargs,
    )
    np.testing.assert_array_equal(mats, expected_mats)
    np.testing.assert_array_equal(rots, expected_rots)
    np.testing.assert_array_equal(result.x, expected_result.x)


@pytest.mark.parametrize(
    "factorize, tensor, indices",
    [
        (
            ffsim.linalg.double_factorized,
            ffsim.random.random_two_body_tensor(3, seed=5678, dtype=float),
            [(1, 0)],
        ),
        (
            ffsim.linalg.double_factorized_t2,
            ffsim.random.random_t2_amplitudes(4, 2, seed=5678, dtype=float),
            [(1, 0)],
        ),
        (
            ffsim.linalg.double_factorized_t2_alpha_beta,
            np.random.default_rng(5678).standard_normal((2, 1, 2, 3)),
            ([(1, 0)], None, None),
        ),
    ],
    ids=["two_body", "t2", "t2_alpha_beta"],
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_double_factorized_optimize_validation(factorize, tensor, indices):
    """Test that the deprecated optimize argument validates like the compression."""
    with pytest.raises(ValueError, match="at least one term"):
        factorize(np.zeros_like(tensor), optimize=True)
    with pytest.raises(ValueError, match="lower triangular"):
        factorize(tensor, optimize=True, diag_coulomb_indices=indices)


def test_double_factorized_compressed_n2_unconstrained():
    """Test compressed double factorization on N2."""
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, -0.5)], ["N", (0, 0, 0.5)]],
        basis="sto-6g",
        symmetry="Dooh",
    )
    n_frozen = 2
    active_space = range(n_frozen, mol.nao_nr())
    scf = pyscf.scf.RHF(mol).run()
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    two_body_tensor = mol_data.hamiltonian.two_body_tensor

    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized(
        two_body_tensor, max_vecs=2
    )
    diag_coulomb_mats_optimized, orbital_rotations_optimized, result = (
        ffsim.linalg.double_factorized_compressed(
            two_body_tensor,
            initial_guess=(diag_coulomb_mats, orbital_rotations),
            options=dict(maxiter=100),
        )
    )
    reconstructed = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    reconstructed_optimal = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations_optimized,
        orbital_rotations_optimized,
        diag_coulomb_mats_optimized,
        orbital_rotations_optimized,
        orbital_rotations_optimized,
    )
    error = np.sum((reconstructed - two_body_tensor) ** 2)
    error_optimized = np.sum((reconstructed_optimal - two_body_tensor) ** 2)
    assert error_optimized < 0.5 * error
    assert np.isrealobj(orbital_rotations_optimized)
    assert np.isrealobj(diag_coulomb_mats_optimized)
    assert result is not None
    assert result.nit <= 100
    assert result.nfev <= 150
    assert result.njev <= 150


def test_double_factorized_compressed_n2_constrained():
    """Test constrained compressed double factorization on N2."""
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, -0.5)], ["N", (0, 0, 0.5)]],
        basis="sto-6g",
        symmetry="Dooh",
    )
    n_frozen = 4
    active_space = range(n_frozen, mol.nao_nr())
    scf = pyscf.scf.RHF(mol).run()
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    norb = mol_data.norb
    two_body_tensor = mol_data.hamiltonian.two_body_tensor

    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized(
        two_body_tensor, max_vecs=2
    )
    diag_coulomb_indices = [(p, p) for p in range(norb)]
    diag_coulomb_indices.extend([(p, p + 1) for p in range(norb - 1)])
    diag_coulomb_indices.extend([(p, p + 2) for p in range(norb - 2)])
    diag_coulomb_mats_optimized, orbital_rotations_optimized, result = (
        ffsim.linalg.double_factorized_compressed(
            two_body_tensor,
            initial_guess=ffsim.linalg.double_factorized(two_body_tensor, max_vecs=4),
            options=dict(maxiter=100),
            diag_coulomb_indices=diag_coulomb_indices,
        )
    )

    diag_coulomb_mask = np.zeros((norb, norb), dtype=bool)
    rows, cols = zip(*diag_coulomb_indices)
    diag_coulomb_mask[rows, cols] = True
    diag_coulomb_mask[cols, rows] = True
    np.testing.assert_array_equal(
        diag_coulomb_mats_optimized, diag_coulomb_mats_optimized * diag_coulomb_mask
    )
    reconstructed = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    reconstructed_optimal = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations_optimized,
        orbital_rotations_optimized,
        diag_coulomb_mats_optimized,
        orbital_rotations_optimized,
        orbital_rotations_optimized,
    )
    error = np.sum((reconstructed - two_body_tensor) ** 2)
    error_optimized = np.sum((reconstructed_optimal - two_body_tensor) ** 2)
    assert error_optimized < 0.5 * error
    assert np.isrealobj(orbital_rotations_optimized)
    assert np.isrealobj(diag_coulomb_mats_optimized)
    assert result is not None
    assert result.nit <= 100
    assert result.nfev <= 120
    assert result.njev <= 120


def test_double_factorized_compressed_random():
    """Test compressed double factorization on random tensor."""
    dim = 2
    two_body_tensor = ffsim.random.random_two_body_tensor(dim, seed=RNG, dtype=float)

    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized(
        two_body_tensor, max_vecs=2
    )
    diag_coulomb_mats_optimized, orbital_rotations_optimized, _ = (
        ffsim.linalg.double_factorized_compressed(
            two_body_tensor, initial_guess=(diag_coulomb_mats, orbital_rotations)
        )
    )
    reconstructed = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    reconstructed_optimal = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations_optimized,
        orbital_rotations_optimized,
        diag_coulomb_mats_optimized,
        orbital_rotations_optimized,
        orbital_rotations_optimized,
    )
    error = np.sum((reconstructed - two_body_tensor) ** 2)
    error_optimized = np.sum((reconstructed_optimal - two_body_tensor) ** 2)
    assert error_optimized < 0.1 * error


@pytest.mark.parametrize("norb", range(2, 6))
def test_double_factorized_compressed_initial_guess(norb: int):
    """Test compressed double factorization starts from the unoptimized result."""
    two_body_tensor = ffsim.random.random_two_body_tensor(norb, seed=RNG, dtype=float)
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized(
        two_body_tensor, cholesky=True
    )
    # The test is only meaningful if some orbital rotations have determinant -1
    assert np.any(np.linalg.det(orbital_rotations) < 0)
    diag_coulomb_mats_optimized, orbital_rotations_optimized, _ = (
        ffsim.linalg.double_factorized_compressed(
            two_body_tensor,
            initial_guess=(diag_coulomb_mats, orbital_rotations),
            options=dict(maxiter=0),
        )
    )
    np.testing.assert_allclose(
        np.linalg.det(orbital_rotations_optimized), 1, atol=1e-12
    )
    reconstructed = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    reconstructed_optimized = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations_optimized,
        orbital_rotations_optimized,
        diag_coulomb_mats_optimized,
        orbital_rotations_optimized,
        orbital_rotations_optimized,
    )
    np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-8)
    np.testing.assert_allclose(reconstructed_optimized, two_body_tensor, atol=1e-8)


def test_double_factorized_compressed_random_constrained():
    """Test constrained compressed double factorization"""
    dim = 3
    two_body_tensor = ffsim.random.random_two_body_tensor(dim, seed=RNG, dtype=float)

    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized(
        two_body_tensor, max_vecs=2
    )
    diag_coulomb_indices = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
    diag_coulomb_mats_optimized, orbital_rotations_optimized, _ = (
        ffsim.linalg.double_factorized_compressed(
            two_body_tensor,
            initial_guess=(diag_coulomb_mats, orbital_rotations),
            diag_coulomb_indices=diag_coulomb_indices,
        )
    )

    diag_coulomb_mask = np.zeros((3, 3), dtype=bool)
    rows, cols = zip(*diag_coulomb_indices)
    diag_coulomb_mask[rows, cols] = True
    diag_coulomb_mask[cols, rows] = True
    np.testing.assert_allclose(
        diag_coulomb_mats_optimized, diag_coulomb_mats_optimized * diag_coulomb_mask
    )
    reconstructed = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    reconstructed_optimal = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations_optimized,
        orbital_rotations_optimized,
        diag_coulomb_mats_optimized,
        orbital_rotations_optimized,
        orbital_rotations_optimized,
    )
    error = np.sum((reconstructed - two_body_tensor) ** 2)
    error_optimized = np.sum((reconstructed_optimal - two_body_tensor) ** 2)
    assert error_optimized < 0.3 * error


@pytest.mark.parametrize("norb, nocc", [(4, 2), (5, 2), (5, 3)])
def test_double_factorized_t2_amplitudes_random(norb: int, nocc: int):
    """Test double factorization of random t2 amplitudes."""
    t2 = ffsim.random.random_t2_amplitudes(norb, nocc, dtype=float)
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2(t2)
    reconstructed = reconstruct_t2(diag_coulomb_mats, orbital_rotations, nocc=nocc)
    np.testing.assert_allclose(reconstructed, t2, atol=1e-8)
    n_reps, _, _ = diag_coulomb_mats.shape
    even_index = list(range(0, n_reps, 2))
    odd_index = list(range(1, n_reps, 2))
    np.testing.assert_allclose(
        diag_coulomb_mats[even_index], -diag_coulomb_mats[odd_index], atol=1e-8
    )
    np.testing.assert_allclose(
        orbital_rotations[even_index], orbital_rotations[odd_index].conj(), atol=1e-8
    )


def test_double_factorized_t2_explicit_norms():
    """Test that term norms are the absolute eigenvalues of the t2 matrix."""
    norb, nocc = 6, 2
    nvrt = norb - nocc
    t2 = ffsim.random.random_t2_amplitudes(norb, nocc, seed=RNG, dtype=float)
    diag_coulomb_mats, orbital_rotations = _double_factorized_t2_explicit(
        t2, max_terms=5
    )
    t2_mat = t2.transpose(0, 2, 1, 3).reshape(nocc * nvrt, nocc * nvrt)
    eigs = np.sort(np.abs(np.linalg.eigvalsh(t2_mat)))[::-1]
    assert len(diag_coulomb_mats) == len(orbital_rotations) == 5
    np.testing.assert_allclose(
        np.linalg.norm(diag_coulomb_mats, axis=(-2, -1)), np.repeat(eigs, 2)[:5]
    )


@pytest.mark.parametrize("cholesky", [False, True])
def test_double_factorized_return_optimize_result_no_optimization(cholesky: bool):
    """Test that None is returned in place of the result when nothing is optimized."""
    norb = 4
    two_body_tensor = ffsim.random.random_two_body_tensor(norb, seed=RNG, dtype=float)
    with pytest.deprecated_call():
        diag_coulomb_mats, orbital_rotations, result = ffsim.linalg.double_factorized(
            two_body_tensor, cholesky=cholesky, return_optimize_result=True
        )
    assert result is None
    expected_mats, expected_rots = ffsim.linalg.double_factorized(
        two_body_tensor, cholesky=cholesky
    )
    np.testing.assert_allclose(diag_coulomb_mats, expected_mats)
    np.testing.assert_allclose(orbital_rotations, expected_rots)


@pytest.mark.parametrize("max_terms", [None, 1])
def test_double_factorized_t2_return_optimize_result_no_optimization(
    max_terms: int | None,
):
    """Test that None is returned in place of the result when nothing is optimized."""
    norb, nocc = 4, 2
    t2 = ffsim.random.random_t2_amplitudes(norb, nocc, seed=RNG, dtype=float)
    with pytest.deprecated_call():
        diag_coulomb_mats, orbital_rotations, result = (
            ffsim.linalg.double_factorized_t2(
                t2, max_terms=max_terms, return_optimize_result=True
            )
        )
    assert result is None
    expected_mats, expected_rots = ffsim.linalg.double_factorized_t2(
        t2, max_terms=max_terms
    )
    np.testing.assert_allclose(diag_coulomb_mats, expected_mats)
    np.testing.assert_allclose(orbital_rotations, expected_rots)

    t2ab = RNG.standard_normal((2, 3, 2, 1))
    with pytest.deprecated_call():
        diag_coulomb_mats, orbital_rotations, result = (
            ffsim.linalg.double_factorized_t2_alpha_beta(
                t2ab, max_terms=max_terms, return_optimize_result=True
            )
        )
    assert result is None
    expected_mats, expected_rots = ffsim.linalg.double_factorized_t2_alpha_beta(
        t2ab, max_terms=max_terms
    )
    np.testing.assert_allclose(diag_coulomb_mats, expected_mats)
    np.testing.assert_allclose(orbital_rotations, expected_rots)


@pytest.mark.parametrize("max_terms", [None, "n_terms", 1000])
def test_double_factorized_t2_compressed_without_truncation(max_terms):
    """Test that compression optimizes even if no terms are truncated."""
    norb, nocc = 4, 2
    t2 = ffsim.random.random_t2_amplitudes(norb, nocc, seed=RNG, dtype=float)
    n_terms = len(ffsim.linalg.double_factorized_t2(t2)[0])
    if max_terms == "n_terms":
        max_terms = n_terms
    diag_coulomb_indices = [(p, p + 1) for p in range(norb - 1)]
    with patch("scipy.optimize.minimize", wraps=scipy.optimize.minimize) as minimize:
        diag_coulomb_mats, orbital_rotations, result = (
            ffsim.linalg.double_factorized_t2_compressed(
                t2,
                max_terms=max_terms,
                diag_coulomb_indices=diag_coulomb_indices,
                options=dict(maxiter=5),
            )
        )
    assert minimize.call_count == 1
    assert isinstance(result, scipy.optimize.OptimizeResult)
    assert len(diag_coulomb_mats) == len(orbital_rotations) == n_terms
    mask = np.zeros((norb, norb), dtype=bool)
    for i, j in diag_coulomb_indices:
        mask[i, j] = mask[j, i] = True
    np.testing.assert_array_equal(diag_coulomb_mats[:, ~mask], 0)

    t2ab = RNG.standard_normal((2, 3, 2, 1))
    n_terms = len(ffsim.linalg.double_factorized_t2_alpha_beta(t2ab)[0])
    diag_coulomb_mats, orbital_rotations, result = (
        ffsim.linalg.double_factorized_t2_alpha_beta_compressed(
            t2ab,
            max_terms=None if max_terms is None else max(max_terms, n_terms),
            options=dict(maxiter=5),
        )
    )
    assert isinstance(result, scipy.optimize.OptimizeResult)
    assert len(diag_coulomb_mats) == len(orbital_rotations) == n_terms


@pytest.mark.parametrize(
    "compress, t2",
    [
        (
            ffsim.linalg.double_factorized_t2_compressed,
            ffsim.random.random_t2_amplitudes(4, 2, seed=3456, dtype=float),
        ),
        (
            ffsim.linalg.double_factorized_t2_alpha_beta_compressed,
            np.random.default_rng(3456).standard_normal((2, 1, 2, 3)),
        ),
    ],
    ids=["t2", "t2_alpha_beta"],
)
@pytest.mark.parametrize(
    "multi_stage_start, multi_stage_step, match",
    [
        (0, None, "multi_stage_start"),
        (-1, None, "multi_stage_start"),
        (None, 0, "multi_stage_step"),
        (None, -1, "multi_stage_step"),
    ],
)
def test_double_factorized_t2_compressed_multi_stage_error(
    compress, t2, multi_stage_start, multi_stage_step, match
):
    """Test that invalid multi-stage arguments raise an error."""
    with pytest.raises(ValueError, match=match):
        compress(
            t2,
            max_terms=2,
            multi_stage_start=multi_stage_start,
            multi_stage_step=multi_stage_step,
        )


@pytest.mark.parametrize(
    "func, explicit, t2",
    [
        (
            ffsim.linalg.double_factorized_t2_compressed,
            "_double_factorized_t2_explicit",
            ffsim.random.random_t2_amplitudes(4, 2, seed=3456, dtype=float),
        ),
        (
            ffsim.linalg.double_factorized_t2_alpha_beta_compressed,
            "_double_factorized_t2_alpha_beta_explicit",
            np.random.default_rng(3456).standard_normal((2, 1, 2, 3)),
        ),
    ],
    ids=["t2", "t2_alpha_beta"],
)
def test_double_factorized_t2_validation_before_factorization(func, explicit, t2):
    """Test that arguments are validated before computing the exact factorization."""
    with patch(f"ffsim.linalg.double_factorized_decomposition.{explicit}") as factorize:
        with pytest.raises(ValueError, match="multi_stage_step"):
            func(t2, max_terms=2, multi_stage_step=0)
        with pytest.raises(ValueError, match="multi_stage_start"):
            func(t2, max_terms=2, multi_stage_start=0)
        with pytest.raises(ValueError, match="complex t2 amplitudes"):
            func(t2.astype(complex))
    factorize.assert_not_called()


def test_double_factorized_t2_amplitudes_max_terms_error():
    """Test double_factorized_t2 raises error on invalid max terms."""
    t2 = ffsim.random.random_t2_amplitudes(4, 2, dtype=float)
    with pytest.raises(ValueError, match="max_terms"):
        _, _ = ffsim.linalg.double_factorized_t2(t2, max_terms=0)


def test_double_factorized_t2_complex_error():
    """Test double_factorized_t2 raises an error on complex amplitudes."""
    t2 = ffsim.random.random_t2_amplitudes(4, 2, dtype=complex)
    with pytest.raises(ValueError, match="complex t2 amplitudes"):
        ffsim.linalg.double_factorized_t2(t2)


def test_double_factorized_t2_tol_max_terms():
    """Test double-factorized decomposition error threshold and max terms."""
    mol = pyscf.gto.Mole()
    mol.build(
        verbose=0,
        atom=[["Li", (0, 0, 0)], ["H", (1.6, 0, 0)]],
        basis="sto-3g",
    )
    hartree_fock = pyscf.scf.RHF(mol)
    hartree_fock.kernel()

    ccsd = pyscf.cc.CCSD(hartree_fock)
    _, _, t2 = ccsd.kernel()
    nocc, _, _, _ = t2.shape

    # test max_vecs
    max_terms = 16
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2(
        t2,
        max_terms=max_terms,
    )
    reconstructed = reconstruct_t2(diag_coulomb_mats, orbital_rotations, nocc=nocc)
    assert len(orbital_rotations) == max_terms
    np.testing.assert_allclose(reconstructed, t2, atol=1e-5)

    # test error threshold
    tol = 1e-3
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2(
        t2, tol=tol
    )
    reconstructed = reconstruct_t2(diag_coulomb_mats, orbital_rotations, nocc=nocc)
    assert len(orbital_rotations) <= 14
    np.testing.assert_allclose(reconstructed, t2, atol=tol)

    # test error threshold and max vecs
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2(
        t2, tol=tol, max_terms=max_terms
    )
    reconstructed = reconstruct_t2(diag_coulomb_mats, orbital_rotations, nocc=nocc)
    assert len(orbital_rotations) <= 14
    np.testing.assert_allclose(reconstructed, t2, atol=tol)


def test_double_factorized_t2_compressed_max_terms_n2_small():
    """Test compressed double factorization for smaller N2."""
    # Build N2 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, -0.5)], ["N", (0, 0, 0.5)]],
        basis="sto-6g",
        symmetry="Dooh",
    )

    # Define active space
    n_frozen = 2
    active_space = range(n_frozen, mol.nao_nr())

    # Get molecular data and Hamiltonian
    scf = pyscf.scf.RHF(mol).run()
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    norb, _ = mol_data.norb, mol_data.nelec

    # Get CCSD t2 amplitudes for initializing the ansatz
    ccsd = pyscf.cc.CCSD(
        scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
    ).run()
    nocc, _, _, _ = ccsd.t2.shape

    # Perform compressed factorization
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]
    diag_coulomb_indices = pairs_aa + pairs_ab
    max_terms = 1
    diag_coulomb_mats_optimized, orbital_rotations_optimized, result = (
        ffsim.linalg.double_factorized_t2_compressed(
            ccsd.t2,
            max_terms=max_terms,
            diag_coulomb_indices=diag_coulomb_indices,
            method="L-BFGS-B",
            options=dict(maxiter=25),
            multi_stage_start=8,
            multi_stage_step=4,
        )
    )
    reconstructed_optimized = (
        1j
        * contract(
            "mpq,map,mip,mbq,mjq->ijab",
            diag_coulomb_mats_optimized,
            orbital_rotations_optimized,
            orbital_rotations_optimized.conj(),
            orbital_rotations_optimized,
            orbital_rotations_optimized.conj(),
        )[:nocc, :nocc, nocc:, nocc:]
    )
    error_optimized = np.sum(np.abs(reconstructed_optimized - ccsd.t2) ** 2)

    # Perform uncompressed factorization
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2(
        ccsd.t2, max_terms=max_terms
    )
    reconstructed = (
        1j
        * contract(
            "mpq,map,mip,mbq,mjq->ijab",
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations.conj(),
            orbital_rotations,
            orbital_rotations.conj(),
        )[:nocc, :nocc, nocc:, nocc:]
    )
    error = np.sum(np.abs(reconstructed - ccsd.t2) ** 2)

    # Check results
    assert error_optimized < 0.7 * error
    assert diag_coulomb_mats_optimized.shape == (max_terms, norb, norb)
    assert orbital_rotations_optimized.shape == (max_terms, norb, norb)
    assert result is not None
    assert result.nit <= 25
    assert result.nfev <= 40
    assert result.njev <= 40


def test_double_factorized_t2_compressed_max_terms_n2_large():
    """Test compressed double factorization for larger N2."""
    # Build N2 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, -0.5)], ["N", (0, 0, 0.5)]],
        basis="6-31g",
        symmetry="Dooh",
    )

    # Define active space
    n_frozen = 2
    active_space = range(n_frozen, mol.nao_nr())

    # Get molecular data and Hamiltonian
    scf = pyscf.scf.RHF(mol).run()
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    norb, _ = mol_data.norb, mol_data.nelec

    # Get CCSD t2 amplitudes for initializing the ansatz
    ccsd = pyscf.cc.CCSD(
        scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
    ).run()
    nocc, _, _, _ = ccsd.t2.shape

    # Perform compressed factorization
    max_terms = 2
    diag_coulomb_indices = [(p, p) for p in range(norb)]
    diag_coulomb_indices.extend([(p, p + 1) for p in range(norb - 1)])
    diag_coulomb_indices.extend([(p, p + 2) for p in range(norb - 2)])
    diag_coulomb_mats_optimized, orbital_rotations_optimized, _ = (
        ffsim.linalg.double_factorized_t2_compressed(
            ccsd.t2,
            max_terms=max_terms,
            method="L-BFGS-B",
            options=dict(maxiter=150),
            diag_coulomb_indices=diag_coulomb_indices,
            regularization=1e-4,
        )
    )
    optimized_diag_coulomb_norm = np.sum(np.abs(diag_coulomb_mats_optimized) ** 2)
    reconstructed_optimized = (
        1j
        * contract(
            "mpq,map,mip,mbq,mjq->ijab",
            diag_coulomb_mats_optimized,
            orbital_rotations_optimized,
            orbital_rotations_optimized.conj(),
            orbital_rotations_optimized,
            orbital_rotations_optimized.conj(),
        )[:nocc, :nocc, nocc:, nocc:]
    )
    error_optimized = np.sum(np.abs(reconstructed_optimized - ccsd.t2) ** 2)

    # Perform uncompressed factorization
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2(
        ccsd.t2, max_terms=max_terms
    )
    init_diag_coulomb_norm = np.sum(np.abs(diag_coulomb_mats) ** 2)
    reconstructed = (
        1j
        * contract(
            "mpq,map,mip,mbq,mjq->ijab",
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations.conj(),
            orbital_rotations,
            orbital_rotations.conj(),
        )[:nocc, :nocc, nocc:, nocc:]
    )
    error = np.sum(np.abs(reconstructed - ccsd.t2) ** 2)

    # Check results
    diag_coulomb_mask = np.zeros((norb, norb), dtype=bool)
    rows, cols = zip(*diag_coulomb_indices)
    diag_coulomb_mask[rows, cols] = True
    diag_coulomb_mask[cols, rows] = True
    np.testing.assert_array_equal(
        diag_coulomb_mats_optimized, diag_coulomb_mats_optimized * diag_coulomb_mask
    )
    assert error_optimized < 0.7 * error
    np.testing.assert_allclose(
        optimized_diag_coulomb_norm, init_diag_coulomb_norm, atol=5
    )
    assert diag_coulomb_mats_optimized.shape == (max_terms, norb, norb)
    assert orbital_rotations_optimized.shape == (max_terms, norb, norb)


def test_double_factorized_t2_compressed_max_terms_random():
    """Test compressed double factorization with random t2"""
    norb = 4
    nocc = 2
    t2 = ffsim.random.random_t2_amplitudes(norb=norb, nocc=nocc, seed=RNG, dtype=float)

    # Perform compressed factorization
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]
    diag_coulomb_indices = pairs_aa + pairs_ab
    max_terms = 2
    diag_coulomb_mats_optimized, orbital_rotations_optimized, _ = (
        ffsim.linalg.double_factorized_t2_compressed(
            t2,
            max_terms=max_terms,
            diag_coulomb_indices=diag_coulomb_indices,
            method="L-BFGS-B",
            options=dict(maxiter=25),
            multi_stage_start=3,
        )
    )
    reconstructed_optimized = (
        1j
        * contract(
            "mpq,map,mip,mbq,mjq->ijab",
            diag_coulomb_mats_optimized,
            orbital_rotations_optimized,
            orbital_rotations_optimized.conj(),
            orbital_rotations_optimized,
            orbital_rotations_optimized.conj(),
        )[:nocc, :nocc, nocc:, nocc:]
    )
    error_optimized = np.sum(np.abs(reconstructed_optimized - t2) ** 2)

    # Perform uncompressed factorization
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2(
        t2, max_terms=max_terms
    )
    reconstructed = (
        1j
        * contract(
            "mpq,map,mip,mbq,mjq->ijab",
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations.conj(),
            orbital_rotations,
            orbital_rotations.conj(),
        )[:nocc, :nocc, nocc:, nocc:]
    )
    error = np.sum(np.abs(reconstructed - t2) ** 2)

    # Check results
    assert diag_coulomb_mats_optimized.shape == (max_terms, norb, norb)
    assert orbital_rotations_optimized.shape == (max_terms, norb, norb)
    assert error_optimized < 0.7 * error


def test_double_factorized_t2_alpha_beta_random():
    """Test double factorization of opposite-spin t2 amplitudes with random tensor."""
    shape = (3, 6, 7, 4)
    t2ab = RNG.standard_normal(shape)
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2_alpha_beta(
        t2ab
    )
    nocc_a, nocc_b, nvrt_a, _ = t2ab.shape
    norb = nocc_a + nvrt_a
    reconstructed = reconstruct_t2_alpha_beta(
        diag_coulomb_mats, orbital_rotations, norb=norb, nocc_a=nocc_a, nocc_b=nocc_b
    )
    np.testing.assert_allclose(reconstructed, t2ab, atol=1e-8)

    n_reps = len(diag_coulomb_mats)
    index_0 = list(range(0, n_reps, 4))
    index_1 = list(range(1, n_reps, 4))
    index_2 = list(range(2, n_reps, 4))
    index_3 = list(range(3, n_reps, 4))

    np.testing.assert_allclose(
        diag_coulomb_mats[index_0], -diag_coulomb_mats[index_1], atol=1e-8
    )
    np.testing.assert_allclose(
        diag_coulomb_mats[index_0], -diag_coulomb_mats[index_2], atol=1e-8
    )
    np.testing.assert_allclose(
        diag_coulomb_mats[index_0], diag_coulomb_mats[index_3], atol=1e-8
    )

    np.testing.assert_allclose(
        orbital_rotations[index_0, 0], orbital_rotations[index_1, 0], atol=1e-8
    )
    np.testing.assert_allclose(
        orbital_rotations[index_0, 0], orbital_rotations[index_2, 0].conj(), atol=1e-8
    )
    np.testing.assert_allclose(
        orbital_rotations[index_0, 0], orbital_rotations[index_3, 0].conj(), atol=1e-8
    )
    # TODO add the rest of the relations


def test_double_factorized_t2_alpha_beta_explicit_norms():
    """Test that spin-orbital term norms are the singular values of the t2 matrix."""
    nocc_a, nocc_b, nvrt_a, nvrt_b = 3, 6, 7, 4
    t2ab = RNG.standard_normal((nocc_a, nocc_b, nvrt_a, nvrt_b))
    diag_coulomb_mats, orbital_rotations = _double_factorized_t2_alpha_beta_explicit(
        t2ab, max_terms=7
    )
    t2_mat = t2ab.transpose(0, 2, 1, 3).reshape(nocc_a * nvrt_a, nocc_b * nvrt_b)
    singular_vals = np.linalg.svd(t2_mat, compute_uv=False)
    assert len(diag_coulomb_mats) == len(orbital_rotations) == 7
    mats_aa, mats_ab, mats_bb = diag_coulomb_mats.transpose(1, 0, 2, 3)
    spin_orbital_mats = np.block(
        [[mats_aa, mats_ab], [mats_ab.transpose(0, 2, 1), mats_bb]]
    )
    np.testing.assert_allclose(
        np.linalg.norm(spin_orbital_mats, axis=(-2, -1)),
        np.repeat(singular_vals, 4)[:7],
    )


def test_double_factorized_t2_alpha_beta_max_terms_error():
    """Test double_factorized_t2_alpha_beta raises error on invalid max terms."""
    shape = (3, 6, 7, 4)
    t2ab = RNG.standard_normal(shape)
    with pytest.raises(ValueError, match="max_terms"):
        _, _ = ffsim.linalg.double_factorized_t2_alpha_beta(t2ab, max_terms=0)


def test_double_factorized_t2_alpha_beta_complex_error():
    """Test double_factorized_t2_alpha_beta raises on complex amplitudes."""
    t2ab = RNG.standard_normal((2, 2, 2, 2)).astype(complex)
    with pytest.raises(ValueError, match="complex t2 amplitudes"):
        ffsim.linalg.double_factorized_t2_alpha_beta(t2ab)


def test_double_factorized_t2_alpha_beta_tol_max_terms():
    """Test double-factorized decomposition alpha-beta error threshold and max terms."""
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["H", (0, 0, 0)], ["O", (0, 0, 1.1)]],
        basis="6-31g",
        spin=1,
        symmetry="Coov",
    )
    hartree_fock = pyscf.scf.ROHF(mol).run()

    ccsd = pyscf.cc.CCSD(hartree_fock).run()
    _, t2ab, _ = ccsd.t2
    nocc_a, nocc_b, nvrt_a, _ = t2ab.shape
    norb = nocc_a + nvrt_a

    # test max_vecs
    max_terms = 100
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2_alpha_beta(
        t2ab, max_terms=max_terms
    )
    reconstructed = reconstruct_t2_alpha_beta(
        diag_coulomb_mats, orbital_rotations, norb=norb, nocc_a=nocc_a, nocc_b=nocc_b
    )
    assert len(diag_coulomb_mats) == max_terms
    np.testing.assert_allclose(reconstructed, t2ab, atol=1e-4)

    # test error threshold
    tol = 1e-3
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2_alpha_beta(
        t2ab, tol=tol
    )
    reconstructed = reconstruct_t2_alpha_beta(
        diag_coulomb_mats, orbital_rotations, norb=norb, nocc_a=nocc_a, nocc_b=nocc_b
    )
    assert len(diag_coulomb_mats) <= 92
    np.testing.assert_allclose(reconstructed, t2ab, atol=tol)

    # test error threshold and max vecs
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2_alpha_beta(
        t2ab, tol=tol, max_terms=max_terms
    )
    reconstructed = reconstruct_t2_alpha_beta(
        diag_coulomb_mats, orbital_rotations, norb=norb, nocc_a=nocc_a, nocc_b=nocc_b
    )
    assert len(orbital_rotations) <= 92
    np.testing.assert_allclose(reconstructed, t2ab, atol=tol)


def test_double_factorized_t2_alpha_beta_compressed_max_terms_h5():
    """Test compressed double factorization for H5."""
    # Build H5 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[("H", (0, 0, i)) for i in range(5)],
        basis="sto-6g",
        spin=1,
        symmetry="Dooh",
        verbose=0,
    )

    # Get molecular data and Hamiltonian
    scf = pyscf.scf.ROHF(mol).run()
    mol_data = ffsim.MolecularData.from_scf(scf)
    norb, _ = mol_data.norb, mol_data.nelec

    # Get CCSD t2 amplitudes for initializing the ansatz
    ccsd = pyscf.cc.CCSD(scf).run()
    _, t2_ab, _ = ccsd.t2
    nocc_a, nocc_b, _, _ = t2_ab.shape

    # Perform compressed factorization
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = pairs_aa + [(p, p) for p in range(norb)]
    max_terms = 1
    diag_coulomb_mats_optimized, orbital_rotations_optimized, result = (
        ffsim.linalg.double_factorized_t2_alpha_beta_compressed(
            t2_ab,
            max_terms=max_terms,
            diag_coulomb_indices=(pairs_aa, pairs_ab, pairs_aa),
            method="L-BFGS-B",
            options=dict(maxiter=25),
            multi_stage_start=8,
            multi_stage_step=4,
        )
    )
    _check_diag_coulomb_alpha_beta_symmetries(diag_coulomb_mats_optimized)
    reconstructed_optimized = reconstruct_t2_alpha_beta(
        diag_coulomb_mats_optimized,
        orbital_rotations_optimized,
        norb=norb,
        nocc_a=nocc_a,
        nocc_b=nocc_b,
    )
    error_optimized = np.sum(np.abs(reconstructed_optimized - t2_ab) ** 2)

    # Perform uncompressed factorization
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2_alpha_beta(
        t2_ab, max_terms=max_terms
    )
    reconstructed = reconstruct_t2_alpha_beta(
        diag_coulomb_mats, orbital_rotations, norb=norb, nocc_a=nocc_a, nocc_b=nocc_b
    )
    error = np.sum(np.abs(reconstructed - t2_ab) ** 2)

    # Check results
    assert error_optimized < 0.7 * error
    assert diag_coulomb_mats_optimized.shape == (max_terms, 3, norb, norb)
    assert orbital_rotations_optimized.shape == (max_terms, 2, norb, norb)
    assert result is not None
    assert result.nit <= 25
    assert result.nfev <= 40
    assert result.njev <= 40


def _check_diag_coulomb_alpha_beta_symmetries(diag_coulomb_mats: np.ndarray):
    mats_aa = diag_coulomb_mats[:, 0]
    mats_bb = diag_coulomb_mats[:, 2]
    np.testing.assert_allclose(mats_aa, mats_aa.transpose(0, 2, 1))
    np.testing.assert_allclose(mats_bb, mats_bb.transpose(0, 2, 1))
