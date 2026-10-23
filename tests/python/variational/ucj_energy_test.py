# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for UCJ energy evaluation by fermionic backpropagation."""

from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import pytest
import scipy.optimize

import ffsim

RNG = np.random.default_rng(4978)

UCJOp = ffsim.UCJOpSpinBalanced | ffsim.UCJOpSpinUnbalanced | ffsim.UCJOpSpinless
Hamiltonian = ffsim.MolecularHamiltonian | ffsim.MolecularHamiltonianSpinless


def statevector_energy(
    ucj_op: UCJOp,
    hamiltonian: Hamiltonian,
    nelec: int | tuple[int, int],
    occupied_orbitals: Any = None,
) -> float:
    """Compute the energy of a UCJ ansatz statevector with respect to a Hamiltonian."""
    norb = ucj_op.norb
    if occupied_orbitals is None:
        occupied_orbitals = (
            range(nelec)
            if isinstance(nelec, int)
            else (range(nelec[0]), range(nelec[1]))
        )
    reference_state = ffsim.slater_determinant(
        norb=norb, occupied_orbitals=occupied_orbitals
    )
    ansatz_state = ffsim.apply_unitary(reference_state, ucj_op, norb=norb, nelec=nelec)
    linop = ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)
    return np.real(np.vdot(ansatz_state, linop @ ansatz_state))


def assert_energy_and_grad_match_statevector(
    ucj_op: UCJOp,
    hamiltonian: Hamiltonian,
    nelec: int | tuple[int, int],
    energy_func: Callable[..., float],
    energy_and_grad_func: Callable[..., Callable[[np.ndarray], Any]],
    *,
    occupied_orbitals: Any = None,
    interaction_pairs: Any = None,
    params: np.ndarray | None = None,
    chunk_size: int = 5,
    grad_indices: Sequence[int] | np.ndarray | None = None,
    rtol: float = 1e-7,
    atol: float = 0.0,
    eps: float = 1e-6,
):
    """Check the energy and gradient against the statevector, with and without chunks.

    The unchunked energy is checked against the statevector energy, and the gradient
    is checked against central finite differences of the statevector energy at
    ``grad_indices``, or at 5 random parameter indices if not given. The chunked
    energy and gradient are checked against the unchunked ones. With norb=4, the
    default ``chunk_size`` of 5 leaves padded terms in both the same-spin and
    opposite-spin two-body blocks.
    """
    expected = statevector_energy(ucj_op, hamiltonian, nelec, occupied_orbitals)
    energy = energy_func(
        ucj_op, hamiltonian, nelec, occupied_orbitals=occupied_orbitals
    )
    np.testing.assert_allclose(energy, expected, rtol=rtol, atol=atol)
    chunked_energy = energy_func(
        ucj_op,
        hamiltonian,
        nelec,
        occupied_orbitals=occupied_orbitals,
        chunk_size=chunk_size,
    )
    np.testing.assert_allclose(chunked_energy, energy, rtol=1e-12, atol=1e-12)

    if params is None:
        params = ucj_op.to_parameters(interaction_pairs=interaction_pairs)
    energy_and_grad_func_kwargs = dict(
        interaction_pairs=interaction_pairs,
        with_final_orbital_rotation=ucj_op.final_orbital_rotation is not None,
        occupied_orbitals=occupied_orbitals,
    )
    value_and_grad = energy_and_grad_func(
        hamiltonian, nelec, **energy_and_grad_func_kwargs
    )
    value, grad = value_and_grad(params)
    np.testing.assert_allclose(value, expected, rtol=rtol, atol=atol)
    assert grad.shape == params.shape
    assert np.all(np.isfinite(grad))

    chunked_value_and_grad = energy_and_grad_func(
        hamiltonian, nelec, **energy_and_grad_func_kwargs, chunk_size=chunk_size
    )
    chunked_value, chunked_grad = chunked_value_and_grad(params)
    np.testing.assert_allclose(chunked_value, value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(chunked_grad, grad, rtol=1e-12, atol=1e-12)

    if grad_indices is None:
        grad_indices = RNG.choice(params.shape[0], size=5, replace=False)
    from_parameters_kwargs = dict(
        norb=ucj_op.norb,
        n_reps=1,
        interaction_pairs=interaction_pairs,
        with_final_orbital_rotation=ucj_op.final_orbital_rotation is not None,
    )
    for index in grad_indices:
        step = np.zeros_like(params)
        step[index] = eps
        plus_op = type(ucj_op).from_parameters(params + step, **from_parameters_kwargs)
        minus_op = type(ucj_op).from_parameters(params - step, **from_parameters_kwargs)
        finite_diff = (
            statevector_energy(plus_op, hamiltonian, nelec, occupied_orbitals)
            - statevector_energy(minus_op, hamiltonian, nelec, occupied_orbitals)
        ) / (2 * eps)
        np.testing.assert_allclose(grad[index], finite_diff, rtol=1e-4, atol=1e-5)


def singular_overlap_orbital_rotation() -> np.ndarray:
    """Return an orbital rotation that can produce singular occupied overlaps."""
    return 0.5 * np.array(
        [
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
        ],
        dtype=complex,
    )


def singular_overlap_diag_coulomb_mat() -> np.ndarray:
    """Return a Jastrow matrix that produces a near-singular overlap phase."""
    return (np.pi / 6) * np.array(
        [
            [0, -1, 1, -2],
            [-1, 0, 1, -2],
            [1, 1, 0, 2],
            [-2, -2, 2, 0],
        ],
        dtype=float,
    )


@pytest.mark.parametrize(
    "nelec, occupied_orbitals, pairs",
    [
        ((2, 2), None, None),
        ((3, 1), None, None),
        ((0, 2), None, None),
        ((2, 1), None, ([], [])),
        ((2, 2), ([0, 3], [3, 1]), ([(0, 0), (1, 1)], None)),
        ((2, 2), ([1, 3], [3, 1]), None),
        ((2, 1), ([1, 3], [2]), ([(0, 1), (1, 2), (2, 3)], [(0, 0), (1, 3)])),
    ],
)
def test_ucj_energy_spin_balanced(nelec, occupied_orbitals, pairs):
    """Test spin-balanced energy and gradient against the statevector."""
    norb = 4
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb,
        n_reps=1,
        interaction_pairs=pairs,
        with_final_orbital_rotation=True,
        seed=RNG,
    )
    assert_energy_and_grad_match_statevector(
        ucj_op,
        mol_hamiltonian,
        nelec,
        ffsim.ucj_energy_spin_balanced,
        ffsim.ucj_energy_and_grad_func_spin_balanced,
        occupied_orbitals=occupied_orbitals,
        interaction_pairs=pairs,
    )


@pytest.mark.parametrize(
    "nelec, occupied_orbitals, pairs",
    [
        ((2, 2), None, None),
        ((3, 1), None, None),
        ((0, 2), None, None),
        ((2, 1), None, ([], [], [])),
        ((2, 2), ([0, 3], [3, 1]), ([(0, 0), (1, 1)], None, None)),
        (
            (2, 1),
            ([1, 3], [2]),
            ([(0, 1), (2, 3)], [(1, 0), (0, 2), (3, 3)], [(0, 0), (1, 2)]),
        ),
    ],
)
def test_ucj_energy_spin_unbalanced(nelec, occupied_orbitals, pairs):
    """Test spin-unbalanced energy and gradient against the statevector."""
    norb = 4
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
    ucj_op = ffsim.random.random_ucj_op_spin_unbalanced(
        norb,
        n_reps=1,
        interaction_pairs=pairs,
        with_final_orbital_rotation=True,
        seed=RNG,
    )
    assert_energy_and_grad_match_statevector(
        ucj_op,
        mol_hamiltonian,
        nelec,
        ffsim.ucj_energy_spin_unbalanced,
        ffsim.ucj_energy_and_grad_func_spin_unbalanced,
        occupied_orbitals=occupied_orbitals,
        interaction_pairs=pairs,
    )


@pytest.mark.parametrize(
    "nelec, occupied_orbitals, pairs",
    [
        (3, None, None),
        (0, None, None),
        (2, None, []),
        (2, [1, 3], None),
        (3, [0, 2, 3], [(0, 1), (1, 3), (2, 2)]),
    ],
)
def test_ucj_energy_spinless(nelec, occupied_orbitals, pairs):
    """Test spinless energy and gradient against the statevector."""
    norb = 4
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian_spinless(norb, seed=RNG)
    ucj_op = ffsim.random.random_ucj_op_spinless(
        norb,
        n_reps=1,
        interaction_pairs=pairs,
        with_final_orbital_rotation=True,
        seed=RNG,
    )
    assert_energy_and_grad_match_statevector(
        ucj_op,
        mol_hamiltonian,
        nelec,
        ffsim.ucj_energy_spinless,
        ffsim.ucj_energy_and_grad_func_spinless,
        occupied_orbitals=occupied_orbitals,
        interaction_pairs=pairs,
    )


def test_ucj_energy_singular_overlap_spin_balanced():
    """Test spin-balanced energy and gradient near a singular overlap."""
    norb = 4
    orbital_rotation = singular_overlap_orbital_rotation()
    diag_coulomb_mat = singular_overlap_diag_coulomb_mat()
    ucj_op = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=np.array(
            [[diag_coulomb_mat, np.zeros_like(diag_coulomb_mat)]]
        ),
        orbital_rotations=orbital_rotation[None],
    )
    params = ucj_op.to_parameters()
    params[0] -= 1e-10
    ucj_op = ffsim.UCJOpSpinBalanced.from_parameters(params, norb=norb, n_reps=1)
    assert_energy_and_grad_match_statevector(
        ucj_op,
        ffsim.random.random_molecular_hamiltonian(norb, seed=0),
        (2, 2),
        ffsim.ucj_energy_spin_balanced,
        ffsim.ucj_energy_and_grad_func_spin_balanced,
        params=params,
        chunk_size=1,
        grad_indices=[0, 5, 18, 26, 30],
        rtol=1e-12,
        atol=1e-12,
    )


def test_ucj_energy_singular_overlap_spin_unbalanced():
    """Test spin-unbalanced energy and gradient near a singular overlap."""
    norb = 4
    orbital_rotation = singular_overlap_orbital_rotation()
    diag_coulomb_mat = singular_overlap_diag_coulomb_mat()
    ucj_op = ffsim.UCJOpSpinUnbalanced(
        diag_coulomb_mats=np.array(
            [[diag_coulomb_mat, np.zeros_like(diag_coulomb_mat), diag_coulomb_mat]]
        ),
        orbital_rotations=np.array([[orbital_rotation, orbital_rotation]]),
    )
    params = ucj_op.to_parameters()
    params[0] -= 1e-10
    ucj_op = ffsim.UCJOpSpinUnbalanced.from_parameters(params, norb=norb, n_reps=1)
    assert_energy_and_grad_match_statevector(
        ucj_op,
        ffsim.random.random_molecular_hamiltonian(norb, seed=0),
        (2, 2),
        ffsim.ucj_energy_spin_unbalanced,
        ffsim.ucj_energy_and_grad_func_spin_unbalanced,
        params=params,
        chunk_size=1,
        grad_indices=[0, 5, 20, 34, 50],
        rtol=1e-12,
        atol=1e-12,
    )


def test_ucj_energy_singular_overlap_spinless():
    """Test spinless energy and gradient near a singular overlap."""
    norb = 4
    orbital_rotation = singular_overlap_orbital_rotation()
    diag_coulomb_mat = singular_overlap_diag_coulomb_mat()
    ucj_op = ffsim.UCJOpSpinless(
        diag_coulomb_mats=diag_coulomb_mat[None],
        orbital_rotations=orbital_rotation[None],
    )
    params = ucj_op.to_parameters()
    params[0] -= 1e-10
    ucj_op = ffsim.UCJOpSpinless.from_parameters(params, norb=norb, n_reps=1)
    assert_energy_and_grad_match_statevector(
        ucj_op,
        ffsim.random.random_molecular_hamiltonian_spinless(norb, seed=0),
        2,
        ffsim.ucj_energy_spinless,
        ffsim.ucj_energy_and_grad_func_spinless,
        params=params,
        chunk_size=1,
        grad_indices=[0, 5, 6, 8, 22],
        rtol=1e-12,
        atol=1e-12,
    )


def test_ucj_energy_and_grad_func_optimize():
    """Test that the energy-and-gradient callable works with scipy.optimize."""
    norb = 3
    nelec = (2, 1)
    pairs = ([(0, 1), (1, 2)], None)
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb,
        n_reps=1,
        interaction_pairs=pairs,
        with_final_orbital_rotation=True,
        diag_coulomb_scale=0.5,
        seed=RNG,
    )
    initial_energy = ffsim.ucj_energy_spin_balanced(ucj_op, mol_hamiltonian, nelec)

    value_and_grad = ffsim.ucj_energy_and_grad_func_spin_balanced(
        mol_hamiltonian,
        nelec,
        interaction_pairs=pairs,
        with_final_orbital_rotation=True,
    )
    result = scipy.optimize.minimize(
        value_and_grad,
        ucj_op.to_parameters(interaction_pairs=pairs),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 5},
    )
    optimized_ucj_op = ffsim.UCJOpSpinBalanced.from_parameters(
        result.x,
        norb=norb,
        n_reps=1,
        interaction_pairs=pairs,
        with_final_orbital_rotation=True,
    )
    optimized_energy = ffsim.ucj_energy_spin_balanced(
        optimized_ucj_op, mol_hamiltonian, nelec
    )
    np.testing.assert_allclose(optimized_energy, result.fun)
    assert optimized_energy < initial_energy


def test_ucj_energy_validation():
    """Test that invalid inputs raise errors."""
    norb = 3
    nelec = (2, 1)
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(norb, n_reps=1, seed=RNG)

    two_layer_op = ffsim.random.random_ucj_op_spin_balanced(norb, n_reps=2, seed=RNG)
    with pytest.raises(NotImplementedError, match="n_reps=1"):
        ffsim.ucj_energy_spin_balanced(two_layer_op, mol_hamiltonian, nelec)

    other_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb + 1, seed=RNG)
    with pytest.raises(ValueError, match="same number of orbitals"):
        ffsim.ucj_energy_spin_balanced(ucj_op, other_hamiltonian, nelec)

    for occupied_orbitals, match in [
        (([0], [0]), "should contain"),
        (([0, 3], [0]), "out of range"),
        (([1, 1], [0]), "duplicates"),
    ]:
        with pytest.raises(ValueError, match=match):
            ffsim.ucj_energy_spin_balanced(
                ucj_op, mol_hamiltonian, nelec, occupied_orbitals=occupied_orbitals
            )

    for chunk_size in [0, -1]:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ffsim.ucj_energy_spin_balanced(
                ucj_op, mol_hamiltonian, nelec, chunk_size=chunk_size
            )
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ffsim.ucj_energy_and_grad_func_spin_balanced(
                mol_hamiltonian, nelec, chunk_size=chunk_size
            )
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ffsim.ucj_energy_and_grad_func_spin_unbalanced(
                mol_hamiltonian, nelec, chunk_size=chunk_size
            )

    value_and_grad = ffsim.ucj_energy_and_grad_func_spin_balanced(
        mol_hamiltonian, nelec
    )
    params = ucj_op.to_parameters()
    with pytest.raises(ValueError, match="did not match"):
        value_and_grad(params[:-1])
    with pytest.raises(ValueError, match="1-dimensional"):
        value_and_grad(params[None])

    for interaction_pairs in [([(0, 1), (0, norb)], None), (None, [(-1, 0)])]:
        with pytest.raises(ValueError, match="outside the range"):
            ffsim.ucj_energy_and_grad_func_spin_balanced(
                mol_hamiltonian, nelec, interaction_pairs=interaction_pairs
            )
    with pytest.raises(ValueError, match="outside the range"):
        ffsim.ucj_energy_and_grad_func_spin_unbalanced(
            mol_hamiltonian, nelec, interaction_pairs=(None, [(norb, 0)], None)
        )

    spinless_op = ffsim.random.random_ucj_op_spinless(norb, n_reps=1, seed=RNG)
    spinless_hamiltonian = ffsim.random.random_molecular_hamiltonian_spinless(
        norb, seed=RNG
    )
    with pytest.raises(ValueError, match="duplicates"):
        ffsim.ucj_energy_spinless(
            spinless_op, spinless_hamiltonian, 2, occupied_orbitals=[1, 1]
        )
    with pytest.raises(ValueError, match="outside the range"):
        ffsim.ucj_energy_and_grad_func_spinless(
            spinless_hamiltonian, 2, interaction_pairs=[(0, norb)]
        )
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        ffsim.ucj_energy_spinless(spinless_op, spinless_hamiltonian, 2, chunk_size=0)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        ffsim.ucj_energy_and_grad_func_spinless(spinless_hamiltonian, 2, chunk_size=0)
