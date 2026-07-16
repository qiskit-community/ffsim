# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for givens decomposition utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.linalg.lapack import zrot

import ffsim
from ffsim.linalg import givens_decomposition

RNG = np.random.default_rng(145192569164181441104242148618648061604)


def reconstruct_orbital_rotation(
    dim: int,
    givens_rotations: list[tuple[float, complex, int, int]],
    phase_shifts: np.ndarray,
) -> np.ndarray:
    """Reconstruct orbital rotation from Givens decomposition."""
    reconstructed = np.eye(dim, dtype=complex)
    for i, phase_shift in enumerate(phase_shifts):
        reconstructed[i] *= phase_shift
    for c, s, i, j in givens_rotations[::-1]:
        reconstructed[:, j], reconstructed[:, i] = zrot(
            reconstructed[:, j], reconstructed[:, i], c, s.conjugate()
        )
    return reconstructed


def unitary_from_antihermitian(dim: int, scale: float, seed) -> np.ndarray:
    """Construct a unitary by exponentiating a scaled random antihermitian matrix."""
    generator = scale * ffsim.random.random_antihermitian(dim, seed=seed)
    return expm(generator)


@pytest.mark.parametrize("dim", range(6))
def test_givens_decomposition_definition(dim: int):
    """Test Givens decomposition definition."""
    for _ in range(3):
        mat = ffsim.random.random_unitary(dim, seed=RNG)
        givens_rotations, phase_shifts = givens_decomposition(mat)
        reconstructed = np.diag(phase_shifts)
        for c, s, i, j in givens_rotations[::-1]:
            givens_mat = np.eye(dim, dtype=complex)
            givens_mat[np.ix_((i, j), (i, j))] = [
                [c, s],
                [-s.conjugate(), c],
            ]
            reconstructed @= givens_mat.conj()
        np.testing.assert_allclose(reconstructed, mat)
        assert len(givens_rotations) == dim * (dim - 1) // 2


@pytest.mark.parametrize("dim", range(6))
def test_givens_decomposition_reconstruct(dim: int):
    """Test Givens decomposition reconstruction of original matrix."""
    for _ in range(3):
        mat = ffsim.random.random_unitary(dim, seed=RNG)
        givens_rotations, phase_shifts = givens_decomposition(mat)
        reconstructed = reconstruct_orbital_rotation(
            dim=dim, givens_rotations=givens_rotations, phase_shifts=phase_shifts
        )
        np.testing.assert_allclose(reconstructed, mat)


@pytest.mark.parametrize("dim", range(6))
@pytest.mark.parametrize("scale", [1e-3, 1e-6, 1e-9, 1e-12, 1e-15])
def test_givens_decomposition_near_identity(dim: int, scale: float):
    """Test Givens decomposition of a near-identity orbital rotation."""
    # Worst case: one elimination per subdiagonal entry of the unitary
    worst_case_length = dim * (dim - 1) // 2
    generator = 1j * scale * ffsim.random.random_hermitian(dim, seed=RNG)
    orbital_rotation = expm(generator)
    tol = 10 * scale
    givens_rotations, phase_shifts = givens_decomposition(orbital_rotation, tol=tol)
    if dim > 1:
        assert len(givens_rotations) < worst_case_length
    reconstructed = reconstruct_orbital_rotation(
        dim=dim, givens_rotations=givens_rotations, phase_shifts=phase_shifts
    )
    np.testing.assert_allclose(reconstructed, orbital_rotation, atol=tol)


@pytest.mark.parametrize("dim", range(6))
def test_givens_decomposition_identity(dim: int):
    """Test Givens decomposition on identity matrix."""
    mat = np.eye(dim)
    givens_rotations, phase_shifts = givens_decomposition(mat)
    assert all(phase_shifts == 1)
    assert len(givens_rotations) == 0


@pytest.mark.parametrize("norb", range(6))
def test_givens_decomposition_no_side_effects(norb: int):
    """Test that the Givens decomposition doesn't modify the original matrix."""
    for _ in range(3):
        mat = ffsim.random.random_unitary(norb, seed=RNG)
        original_mat = mat.copy()
        _ = givens_decomposition(mat)

        assert ffsim.linalg.is_unitary(original_mat)
        assert ffsim.linalg.is_unitary(mat)
        np.testing.assert_allclose(mat, original_mat, atol=1e-12)


def test_givens_decomposition_no_side_effects_special_case():
    """Test that the Givens decomposition doesn't modify the original matrix."""
    datadir = Path(__file__).parent.parent / "test_data"
    filepath = datadir / "orbital_rotation-0.npy"

    with open(filepath, "rb") as f:
        mat = np.load(f)
    assert ffsim.linalg.is_unitary(mat, atol=1e-12)

    original_mat = mat.copy()
    _ = givens_decomposition(mat)

    assert ffsim.linalg.is_unitary(original_mat)
    assert ffsim.linalg.is_unitary(mat)
    np.testing.assert_allclose(mat, original_mat, atol=1e-12)


@pytest.mark.parametrize("dim", range(6))
def test_givens_decomposition_compressed_exact(dim: int):
    """Test that a non-binding cap reproduces the exact decomposition."""
    max_full = dim * (dim - 1) // 2
    for max_givens, max_layers in [(max_full, None), (None, dim), (None, None)]:
        mat = unitary_from_antihermitian(dim, 1.0, seed=RNG)
        givens_rotations, phase_shifts = givens_decomposition(
            mat, max_givens=max_givens, max_layers=max_layers
        )
        assert len(givens_rotations) == max_full
        reconstructed = reconstruct_orbital_rotation(
            dim=dim, givens_rotations=givens_rotations, phase_shifts=phase_shifts
        )
        np.testing.assert_allclose(reconstructed, mat, atol=1e-12)


def test_givens_decomposition_compressed_max_givens():
    """Test compressing to a maximum number of Givens rotations."""
    dim = 5
    max_full = dim * (dim - 1) // 2
    scale = 0.1
    mat = unitary_from_antihermitian(dim, scale, seed=RNG)
    for max_givens, tol in [
        (max_full // 2, 7e-1),
        (max_full - 1, 2e-1),
        (max_full, 1e-8),
    ]:
        givens_rotations, phase_shifts = givens_decomposition(
            mat, max_givens=max_givens
        )
        assert len(givens_rotations) == max_givens
        # Every rotation acts on adjacent indices
        assert all(abs(i - j) == 1 for _, _, i, j in givens_rotations)
        reconstructed = reconstruct_orbital_rotation(
            dim=dim, givens_rotations=givens_rotations, phase_shifts=phase_shifts
        )
        error = np.linalg.norm(mat - reconstructed)
        assert error <= tol


def test_givens_decomposition_compressed_max_layers():
    """Test compressing to a maximum number of brickwork layers."""
    dim = 5
    scale = 0.1
    mat = unitary_from_antihermitian(dim, scale, seed=RNG)
    for max_layers, tol in [
        (dim // 2, 8e-1),
        (dim - 1, 5e-1),
        (dim, 1e-8),
    ]:
        givens_rotations, phase_shifts = givens_decomposition(
            mat, max_layers=max_layers
        )
        assert all(abs(i - j) == 1 for _, _, i, j in givens_rotations)
        reconstructed = reconstruct_orbital_rotation(
            dim=dim, givens_rotations=givens_rotations, phase_shifts=phase_shifts
        )
        error = np.linalg.norm(mat - reconstructed)
        assert error <= tol


def test_givens_decomposition_compressed_both_caps():
    """Test that both caps combine to the tighter (intersection) constraint."""
    from ffsim.linalg.givens import _brickwork_givens_rotations

    dim = 5
    mat = unitary_from_antihermitian(dim, 1.0, seed=RNG)

    # Determine how many gates lie in the first max_layers layers.
    givens_rotations, _ = givens_decomposition(mat, tol=0.0)
    pairs = [(i, j) for _, _, i, j in givens_rotations]
    thetas = [0.0] * len(pairs)
    phis = [0.0] * len(pairs)
    _, _, _, layer_ids = _brickwork_givens_rotations(pairs, thetas, phis, norb=dim)

    max_layers = 3
    gates_in_layers = sum(1 for layer_id in layer_ids if layer_id < max_layers)

    # max_givens tighter than max_layers
    max_givens = gates_in_layers - 2
    rotations, _ = givens_decomposition(
        mat, max_givens=max_givens, max_layers=max_layers
    )
    assert len(rotations) == min(max_givens, gates_in_layers)

    # max_layers tighter than max_givens
    max_givens = gates_in_layers + 2
    rotations, _ = givens_decomposition(
        mat, max_givens=max_givens, max_layers=max_layers
    )
    assert len(rotations) == min(max_givens, gates_in_layers)


def test_givens_decomposition_compressed_near_identity_returns_fewer():
    """A near-identity rotation must not be padded up to max_givens."""
    dim = 5
    max_full = dim * (dim - 1) // 2
    scale = 1e-9
    tol = 10 * scale
    generator = 1j * scale * ffsim.random.random_hermitian(dim, seed=RNG)
    mat = expm(generator)

    # Exact tol-respecting decomposition uses far fewer than max_full rotations.
    exact_rotations, _ = givens_decomposition(mat, tol=tol)
    n_existing = len(exact_rotations)
    assert n_existing < max_full

    # A budget larger than n_existing must NOT be padded up to the budget.
    givens_rotations, phase_shifts = givens_decomposition(
        mat, tol=tol, max_givens=max_full
    )
    assert len(givens_rotations) == n_existing
    reconstructed = reconstruct_orbital_rotation(
        dim=dim, givens_rotations=givens_rotations, phase_shifts=phase_shifts
    )
    np.testing.assert_allclose(reconstructed, mat, atol=tol)


def test_givens_decomposition_compressed_near_identity_trims_when_binding():
    """When the budget is below the exact count, trim to exactly the budget."""
    dim = 5
    scale = 1e-3
    tol = 10 * scale
    generator = 1j * scale * ffsim.random.random_hermitian(dim, seed=RNG)
    mat = expm(generator)

    exact_rotations, _ = givens_decomposition(mat, tol=tol)
    n_existing = len(exact_rotations)
    if n_existing:
        max_givens = n_existing - 1
        givens_rotations, _ = givens_decomposition(mat, tol=tol, max_givens=max_givens)
        assert len(givens_rotations) == max_givens
        # Every rotation acts on adjacent indices
        assert all(abs(i - j) == 1 for _, _, i, j in givens_rotations)


def test_givens_decomposition_compressed_negative_cap():
    """Test that negative caps raise an error."""
    mat = unitary_from_antihermitian(4, 1.0, seed=RNG)
    with pytest.raises(ValueError, match="max_givens"):
        givens_decomposition(mat, max_givens=-1)
    with pytest.raises(ValueError, match="max_layers"):
        givens_decomposition(mat, max_layers=-1)
