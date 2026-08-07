"""Implement the fermionic backpropagation algorithm for UCJ energy calculation."""

from __future__ import annotations

import functools
import itertools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

from ffsim.hamiltonians.molecular_hamiltonian import (
    MolecularHamiltonian,
    MolecularHamiltonianSpinless,
)
from ffsim.linalg.util import (
    real_symmetrics_from_parameters_jax,
    unitary_from_parameters_jax,
)
from ffsim.variational.ucj_spin_balanced import UCJOpSpinBalanced
from ffsim.variational.ucj_spin_unbalanced import UCJOpSpinUnbalanced
from ffsim.variational.ucj_spinless import UCJOpSpinless
from ffsim.variational.util import validate_interaction_pairs

jax.config.update("jax_enable_x64", True)


def ucj_energy(
    ucj_op: UCJOpSpinBalanced | UCJOpSpinUnbalanced | UCJOpSpinless,
    hamiltonian: MolecularHamiltonian | MolecularHamiltonianSpinless,
    nelec: int | tuple[int, int],
    interaction_pairs=None,
    *,
    occupied_orbitals=None,
    chunk_size: int | None = None,
) -> float:
    """Convenience dispatcher for UCJ energy calculation."""

    if isinstance(ucj_op, UCJOpSpinBalanced) and isinstance(
        hamiltonian, MolecularHamiltonian
    ):
        if not isinstance(nelec, tuple):
            raise TypeError("nelec must be a tuple for a spin-balanced UCJ operator.")
        return ucj_energy_spin_balanced(
            ucj_op,
            hamiltonian,
            nelec,
            interaction_pairs,
            occupied_orbitals=occupied_orbitals,
            chunk_size=chunk_size,
        )
    if isinstance(ucj_op, UCJOpSpinUnbalanced) and isinstance(
        hamiltonian, MolecularHamiltonian
    ):
        if not isinstance(nelec, tuple):
            raise TypeError("nelec must be a tuple for a spin-unbalanced UCJ operator.")
        return ucj_energy_spin_unbalanced(
            ucj_op,
            hamiltonian,
            nelec,
            interaction_pairs,
            occupied_orbitals=occupied_orbitals,
            chunk_size=chunk_size,
        )
    if isinstance(ucj_op, UCJOpSpinless) and isinstance(
        hamiltonian, MolecularHamiltonianSpinless
    ):
        if not isinstance(nelec, int):
            raise TypeError("nelec must be an int for a spinless UCJ operator.")
        return ucj_energy_spinless(
            ucj_op,
            hamiltonian,
            nelec,
            occupied_orbitals=occupied_orbitals,
            chunk_size=chunk_size,
        )
    raise TypeError(f"Unsupported UCJ operator type: {type(ucj_op)}")


def ucj_energy_and_grad(
    ucj_op: UCJOpSpinBalanced | UCJOpSpinUnbalanced | UCJOpSpinless,
    hamiltonian: MolecularHamiltonian | MolecularHamiltonianSpinless,
    nelec: int | tuple[int, int],
    interaction_pairs=None,
    *,
    occupied_orbitals=None,
    chunk_size: int | None = None,
) -> tuple[float, np.ndarray]:
    """Convenience dispatcher for UCJ energy and gradient calculation."""

    if isinstance(ucj_op, UCJOpSpinBalanced) and isinstance(
        hamiltonian, MolecularHamiltonian
    ):
        if not isinstance(nelec, tuple):
            raise TypeError("nelec must be a tuple for a spin-balanced UCJ operator.")
        return ucj_energy_and_grad_spin_balanced(
            ucj_op,
            hamiltonian,
            nelec,
            interaction_pairs,
            occupied_orbitals=occupied_orbitals,
            chunk_size=chunk_size,
        )
    if isinstance(ucj_op, UCJOpSpinUnbalanced) and isinstance(
        hamiltonian, MolecularHamiltonian
    ):
        if not isinstance(nelec, tuple):
            raise TypeError("nelec must be a tuple for a spin-unbalanced UCJ operator.")
        return ucj_energy_and_grad_spin_unbalanced(
            ucj_op,
            hamiltonian,
            nelec,
            interaction_pairs,
            occupied_orbitals=occupied_orbitals,
            chunk_size=chunk_size,
        )
    if isinstance(ucj_op, UCJOpSpinless) and isinstance(
        hamiltonian, MolecularHamiltonianSpinless
    ):
        if not isinstance(nelec, int):
            raise TypeError("nelec must be an int for a spinless UCJ operator.")
        return ucj_energy_and_grad_spinless(
            ucj_op,
            hamiltonian,
            nelec,
            interaction_pairs,
            occupied_orbitals=occupied_orbitals,
            chunk_size=chunk_size,
        )
    raise TypeError(f"Unsupported UCJ operator type: {type(ucj_op)}")


def optimize_ucj_energy(
    initial_ucj_op: UCJOpSpinBalanced | UCJOpSpinUnbalanced | UCJOpSpinless,
    hamiltonian: MolecularHamiltonian | MolecularHamiltonianSpinless,
    nelec: int | tuple[int, int],
    *,
    interaction_pairs=None,
    occupied_orbitals=None,
    chunk_size: int | None = None,
    method: str = "L-BFGS-B",
    callback=None,
    options: dict | None = None,
    return_optimize_result: bool = False,
) -> (
    UCJOpSpinBalanced
    | UCJOpSpinUnbalanced
    | UCJOpSpinless
    | tuple[
        UCJOpSpinBalanced | UCJOpSpinUnbalanced | UCJOpSpinless,
        scipy.optimize.OptimizeResult,
    ]
):
    """Convenience dispatcher for UCJ energy optimization."""

    if isinstance(initial_ucj_op, UCJOpSpinBalanced) and isinstance(
        hamiltonian, MolecularHamiltonian
    ):
        if not isinstance(nelec, tuple):
            raise TypeError("nelec must be a tuple for a spin-balanced UCJ operator.")
        return optimize_ucj_energy_spin_balanced(
            initial_ucj_op,
            hamiltonian,
            nelec,
            interaction_pairs=interaction_pairs,
            occupied_orbitals=occupied_orbitals,
            chunk_size=chunk_size,
            method=method,
            callback=callback,
            options=options,
            return_optimize_result=return_optimize_result,
        )

    if isinstance(initial_ucj_op, UCJOpSpinUnbalanced) and isinstance(
        hamiltonian, MolecularHamiltonian
    ):
        if not isinstance(nelec, tuple):
            raise TypeError("nelec must be a tuple for a spin-unbalanced UCJ operator.")
        return optimize_ucj_energy_spin_unbalanced(
            initial_ucj_op,
            hamiltonian,
            nelec,
            interaction_pairs=interaction_pairs,
            occupied_orbitals=occupied_orbitals,
            chunk_size=chunk_size,
            method=method,
            callback=callback,
            options=options,
            return_optimize_result=return_optimize_result,
        )

    if isinstance(initial_ucj_op, UCJOpSpinless) and isinstance(
        hamiltonian, MolecularHamiltonianSpinless
    ):
        if not isinstance(nelec, int):
            raise TypeError("nelec must be an int for a spinless UCJ operator.")
        return optimize_ucj_energy_spinless(
            initial_ucj_op,
            hamiltonian,
            nelec,
            interaction_pairs=interaction_pairs,
            occupied_orbitals=occupied_orbitals,
            chunk_size=chunk_size,
            method=method,
            callback=callback,
            options=options,
            return_optimize_result=return_optimize_result,
        )

    raise TypeError(f"Unsupported UCJ operator type: {type(initial_ucj_op)}")


def ucj_energy_spin_balanced(
    ucj_op: UCJOpSpinBalanced,
    hamiltonian: MolecularHamiltonian,
    nelec: tuple[int, int],
    interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]
    | None = None,
    *,
    occupied_orbitals: tuple[Sequence[int], Sequence[int]] | None = None,
    chunk_size: int | None = None,
) -> float:
    """Compute the UCJ energy for a spin-balanced system
    using the fermionic backpropagation outlined in https://arxiv.org/abs/2607.21337.

    Args:
        hamiltonian: The Hamiltonian.
        ucj_op: The UCJ operator. Must have n_reps=1, with an optional final
            orbital rotation.
        nelec: The number of alpha and beta electrons.
        interaction_pairs: The interaction pairs to consider. If None, all pairs are
            considered.
        occupied_orbitals: The occupied orbitals for the reference state. Defaults to
            the Hartree-Fock state.
        chunk_size: The number of two-body Hamiltonian terms to process at a time.
            If ``None``, all two-body terms are processed in one batch. This is useful
            for large systems where the two-body tensor may not fit in GPU memory.

    Returns:
        The expectation value of the Hamiltonian with respect to the UCJ state.
    """

    _validate_ucj_op(ucj_op)
    _validate_molecular_hamiltonian(hamiltonian, ucj_op.norb)

    pairs_aa, pairs_ab = (
        interaction_pairs if interaction_pairs is not None else (None, None)
    )
    validate_interaction_pairs(pairs_aa, ordered=False)
    validate_interaction_pairs(pairs_ab, ordered=False)

    norb = ucj_op.norb
    diag_coulomb_mats = ucj_op.diag_coulomb_mats
    orbital_rotation = ucj_op.orbital_rotations[0]
    final_orbital_rotation = ucj_op.final_orbital_rotation

    # Compute :math:`u = e^{-K}` for the combined orbital rotation.
    u = (
        orbital_rotation
        if final_orbital_rotation is None
        else final_orbital_rotation @ orbital_rotation
    )

    occupied_alpha, occupied_beta = _occupied_orbitals_key(
        norb, nelec, occupied_orbitals
    )
    rotated_reference = orbital_rotation.conj().T
    q_alpha, q_beta = (
        jnp.asarray(rotated_reference[:, occupied_alpha]),
        jnp.asarray(rotated_reference[:, occupied_beta]),
    )

    h_pq, g_pqrs = _propagate_through_orbital_rotations(
        jnp.asarray(hamiltonian.one_body_tensor),
        jnp.asarray(hamiltonian.two_body_tensor),
        jnp.asarray(u),
    )

    jastrow_mat, jastrow_vec = _spin_balanced_jastrow_phase(
        jnp.asarray(diag_coulomb_mats[0][0]),
        jnp.asarray(diag_coulomb_mats[0][1]),
        norb,
    )
    return float(
        _compute_energy_spin_balanced(
            q_alpha,
            q_beta,
            jnp.asarray(hamiltonian.constant),
            h_pq,
            g_pqrs,
            jastrow_mat,
            jastrow_vec,
            norb,
            chunk_size=chunk_size,
        )
    )


def ucj_energy_and_grad_spin_balanced(
    ucj_op: UCJOpSpinBalanced,
    hamiltonian: MolecularHamiltonian,
    nelec: tuple[int, int],
    interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]
    | None = None,
    *,
    occupied_orbitals: tuple[Sequence[int], Sequence[int]] | None = None,
    chunk_size: int | None = None,
) -> tuple[float, np.ndarray]:
    """Compute the UCJ energy and parameter gradient for a spin-balanced system.

    The gradient is with respect to the flattened parameter vector returned by
    ``ucj_op.to_parameters(interaction_pairs=interaction_pairs)``.

    Args:
        hamiltonian: The Hamiltonian.
        ucj_op: The UCJ operator. Must have n_reps=1, with an optional final
            orbital rotation.
        nelec: The number of alpha and beta electrons.
        interaction_pairs: The interaction pairs used to parameterize the Jastrow
            matrices. If None, all pairs are considered.
        occupied_orbitals: The occupied orbitals for the reference state. Defaults to
            the Hartree-Fock state.
        chunk_size: The number of two-body Hamiltonian terms to process at a time.
            If ``None``, all two-body terms are processed in one batch.

    Returns:
        The energy and its gradient with respect to the UCJ parameter vector.
    """
    _validate_ucj_op(ucj_op)
    _validate_molecular_hamiltonian(hamiltonian, ucj_op.norb)

    norb = ucj_op.norb
    with_final_orbital_rotation = ucj_op.final_orbital_rotation is not None
    triu_indices = list(itertools.combinations_with_replacement(range(norb), 2))
    pairs_aa, pairs_ab = (
        interaction_pairs if interaction_pairs is not None else (None, None)
    )
    validate_interaction_pairs(pairs_aa, ordered=False)
    validate_interaction_pairs(pairs_ab, ordered=False)
    pairs_aa = triu_indices if pairs_aa is None else pairs_aa
    pairs_ab = triu_indices if pairs_ab is None else pairs_ab
    interaction_pairs_key = (
        _interaction_pairs_key(pairs_aa),
        _interaction_pairs_key(pairs_ab),
    )
    occupied_orbitals_key = _occupied_orbitals_key(norb, nelec, occupied_orbitals)

    value_and_grad = _make_spin_balanced_objective(
        norb,
        interaction_pairs_key,
        with_final_orbital_rotation,
        occupied_orbitals_key,
        chunk_size,
    )
    return _evaluate_ucj_value_and_grad(
        value_and_grad,
        ucj_op.to_parameters(interaction_pairs=interaction_pairs),
        hamiltonian,
    )


def optimize_ucj_energy_spin_balanced(
    initial_ucj_op: UCJOpSpinBalanced,
    hamiltonian: MolecularHamiltonian,
    nelec: tuple[int, int],
    *,
    interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]
    | None = None,
    occupied_orbitals: tuple[Sequence[int], Sequence[int]] | None = None,
    chunk_size: int | None = None,
    method: str = "L-BFGS-B",
    callback=None,
    options: dict | None = None,
    return_optimize_result: bool = False,
) -> UCJOpSpinBalanced | tuple[UCJOpSpinBalanced, scipy.optimize.OptimizeResult]:
    """Optimize a spin-balanced UCJ operator to minimize energy.

    Args:
        initial_ucj_op: The initial UCJ operator. Must have n_reps=1, with an
            optional final orbital rotation.
        hamiltonian: The Hamiltonian.
        nelec: The number of alpha and beta electrons.
        interaction_pairs: The interaction pairs used to parameterize the Jastrow
            matrices. If None, all pairs are considered.
        occupied_orbitals: The occupied orbitals for the reference state. Defaults to
            the Hartree-Fock state.
        chunk_size: The number of two-body Hamiltonian terms to process at a time.
            If ``None``, all two-body terms are processed in one batch.
        method: The optimization method. See the documentation of
            `scipy.optimize.minimize`_ for possible values.
        callback: Callback function for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
        options: Options for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
        return_optimize_result: Whether to also return the `OptimizeResult`_ returned
            by `scipy.optimize.minimize`_.

    Returns:
        The optimized UCJ operator. If ``return_optimize_result`` is set to ``True``,
        the `OptimizeResult`_ returned by `scipy.optimize.minimize`_ is also returned.

    .. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    .. _OptimizeResult: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    _validate_ucj_op(initial_ucj_op)
    _validate_molecular_hamiltonian(hamiltonian, initial_ucj_op.norb)

    norb = initial_ucj_op.norb
    with_final_orbital_rotation = initial_ucj_op.final_orbital_rotation is not None
    triu_indices = list(itertools.combinations_with_replacement(range(norb), 2))
    pairs_aa, pairs_ab = (
        interaction_pairs if interaction_pairs is not None else (None, None)
    )
    validate_interaction_pairs(pairs_aa, ordered=False)
    validate_interaction_pairs(pairs_ab, ordered=False)
    pairs_aa = triu_indices if pairs_aa is None else pairs_aa
    pairs_ab = triu_indices if pairs_ab is None else pairs_ab
    interaction_pairs_key = (
        _interaction_pairs_key(pairs_aa),
        _interaction_pairs_key(pairs_ab),
    )
    occupied_orbitals_key = _occupied_orbitals_key(norb, nelec, occupied_orbitals)

    value_and_grad = _make_spin_balanced_objective(
        norb,
        interaction_pairs_key,
        with_final_orbital_rotation,
        occupied_orbitals_key,
        chunk_size,
    )
    result = _minimize_ucj_parameters(
        value_and_grad,
        initial_ucj_op.to_parameters(interaction_pairs=interaction_pairs),
        hamiltonian,
        method,
        callback,
        options,
    )
    optimized_ucj_op = UCJOpSpinBalanced.from_parameters(
        result.x,
        norb=initial_ucj_op.norb,
        n_reps=1,
        interaction_pairs=interaction_pairs,
        with_final_orbital_rotation=with_final_orbital_rotation,
    )

    if return_optimize_result:
        return optimized_ucj_op, result

    return optimized_ucj_op


def ucj_energy_spin_unbalanced(
    ucj_op: UCJOpSpinUnbalanced,
    hamiltonian: MolecularHamiltonian,
    nelec: tuple[int, int],
    interaction_pairs: tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ]
    | None = None,
    *,
    occupied_orbitals: tuple[Sequence[int], Sequence[int]] | None = None,
    chunk_size: int | None = None,
) -> float:
    """Compute the UCJ energy for a spin-unbalanced system
    using the fermionic backpropagation outlined in https://arxiv.org/abs/2607.21337.

    Args:
        hamiltonian: The Hamiltonian.
        ucj_op: The UCJ operator. Must have n_reps=1, with an optional final
            orbital rotation.
        nelec: The number of alpha and beta electrons.
        interaction_pairs: The interaction pairs to consider. If None, all pairs are
            considered.
        occupied_orbitals: The occupied orbitals for the reference state. Defaults to
            the Hartree-Fock state.

    Returns:
        The expectation value of the Hamiltonian with respect to the UCJ state.
    """
    _validate_ucj_op(ucj_op)
    _validate_molecular_hamiltonian(hamiltonian, ucj_op.norb)

    pairs_aa, pairs_ab, pairs_bb = (
        interaction_pairs if interaction_pairs is not None else (None, None, None)
    )
    validate_interaction_pairs(pairs_aa, ordered=False)
    validate_interaction_pairs(pairs_ab, ordered=True)
    validate_interaction_pairs(pairs_bb, ordered=False)

    norb = ucj_op.norb
    diag_coulomb_mats = ucj_op.diag_coulomb_mats
    orbital_rotation_alpha, orbital_rotation_beta = ucj_op.orbital_rotations[0]
    final_orbital_rotation = ucj_op.final_orbital_rotation

    u_alpha = (
        orbital_rotation_alpha
        if final_orbital_rotation is None
        else final_orbital_rotation[0] @ orbital_rotation_alpha
    )
    u_beta = (
        orbital_rotation_beta
        if final_orbital_rotation is None
        else final_orbital_rotation[1] @ orbital_rotation_beta
    )

    occupied_alpha, occupied_beta = _occupied_orbitals_key(
        norb, nelec, occupied_orbitals
    )
    q_alpha, q_beta = (
        jnp.asarray(orbital_rotation_alpha.conj().T[:, occupied_alpha]),
        jnp.asarray(orbital_rotation_beta.conj().T[:, occupied_beta]),
    )

    one_body_tensor = jnp.asarray(hamiltonian.one_body_tensor)
    two_body_tensor = jnp.asarray(hamiltonian.two_body_tensor)
    h_alpha, g_aa = _propagate_through_orbital_rotations(
        one_body_tensor, two_body_tensor, jnp.asarray(u_alpha)
    )
    h_beta, g_bb = _propagate_through_orbital_rotations(
        one_body_tensor, two_body_tensor, jnp.asarray(u_beta)
    )
    g_ab = _propagate_spin_sector_tensor(
        two_body_tensor, jnp.asarray(u_alpha), jnp.asarray(u_beta)
    )
    g_ba = _propagate_spin_sector_tensor(
        two_body_tensor, jnp.asarray(u_beta), jnp.asarray(u_alpha)
    )

    jastrow_mat, jastrow_vec = _spin_unbalanced_jastrow_phase(
        jnp.asarray(diag_coulomb_mats[0][0]),
        jnp.asarray(diag_coulomb_mats[0][1]),
        jnp.asarray(diag_coulomb_mats[0][2]),
        norb,
    )
    return float(
        _compute_energy_spin_unbalanced(
            q_alpha,
            q_beta,
            jnp.asarray(hamiltonian.constant),
            h_alpha,
            h_beta,
            g_aa,
            g_ab,
            g_ba,
            g_bb,
            jastrow_mat,
            jastrow_vec,
            norb,
            chunk_size=chunk_size,
        )
    )


def ucj_energy_and_grad_spin_unbalanced(
    ucj_op: UCJOpSpinUnbalanced,
    hamiltonian: MolecularHamiltonian,
    nelec: tuple[int, int],
    interaction_pairs: tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ]
    | None = None,
    *,
    occupied_orbitals: tuple[Sequence[int], Sequence[int]] | None = None,
    chunk_size: int | None = None,
) -> tuple[float, np.ndarray]:
    """Compute the UCJ energy and parameter gradient for a spin-unbalanced system.

    The gradient is with respect to the flattened parameter vector returned by
    ``ucj_op.to_parameters(interaction_pairs=interaction_pairs)``.

    Args:
        hamiltonian: The Hamiltonian.
        ucj_op: The UCJ operator. Must have n_reps=1, with an optional final
            orbital rotation.
        nelec: The number of alpha and beta electrons.
        interaction_pairs: The interaction pairs used to parameterize the Jastrow
            matrices. If None, all pairs are considered.
        occupied_orbitals: The occupied orbitals for the reference state. Defaults to
            the Hartree-Fock state.
        chunk_size: The number of two-body Hamiltonian terms to process at a time.
            If ``None``, all two-body terms are processed in one batch.

    Returns:
        The energy and its gradient with respect to the UCJ parameter vector.
    """
    _validate_ucj_op(ucj_op)
    _validate_molecular_hamiltonian(hamiltonian, ucj_op.norb)

    norb = ucj_op.norb
    with_final_orbital_rotation = ucj_op.final_orbital_rotation is not None
    triu_indices = list(itertools.combinations_with_replacement(range(norb), 2))
    mat_indices = list(itertools.product(range(norb), repeat=2))
    pairs_aa, pairs_ab, pairs_bb = (
        interaction_pairs if interaction_pairs is not None else (None, None, None)
    )
    validate_interaction_pairs(pairs_aa, ordered=False)
    validate_interaction_pairs(pairs_ab, ordered=True)
    validate_interaction_pairs(pairs_bb, ordered=False)
    pairs_aa = triu_indices if pairs_aa is None else pairs_aa
    pairs_ab = mat_indices if pairs_ab is None else pairs_ab
    pairs_bb = triu_indices if pairs_bb is None else pairs_bb
    interaction_pairs_key = (
        _interaction_pairs_key(pairs_aa),
        _interaction_pairs_key(pairs_ab),
        _interaction_pairs_key(pairs_bb),
    )
    occupied_orbitals_key = _occupied_orbitals_key(norb, nelec, occupied_orbitals)

    value_and_grad = _make_spin_unbalanced_objective(
        norb,
        interaction_pairs_key,
        with_final_orbital_rotation,
        occupied_orbitals_key,
        chunk_size,
    )
    return _evaluate_ucj_value_and_grad(
        value_and_grad,
        ucj_op.to_parameters(interaction_pairs),
        hamiltonian,
    )


def optimize_ucj_energy_spin_unbalanced(
    initial_ucj_op: UCJOpSpinUnbalanced,
    hamiltonian: MolecularHamiltonian,
    nelec: tuple[int, int],
    *,
    interaction_pairs: tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ]
    | None = None,
    occupied_orbitals: tuple[Sequence[int], Sequence[int]] | None = None,
    chunk_size: int | None = None,
    method: str = "L-BFGS-B",
    callback=None,
    options: dict | None = None,
    return_optimize_result: bool = False,
) -> UCJOpSpinUnbalanced | tuple[UCJOpSpinUnbalanced, scipy.optimize.OptimizeResult]:
    """Optimize a spin-unbalanced UCJ operator to minimize energy.

    Args:
        initial_ucj_op: The initial UCJ operator. Must have n_reps=1, with an
            optional final orbital rotation.
        hamiltonian: The Hamiltonian.
        nelec: The number of alpha and beta electrons.
        interaction_pairs: The interaction pairs used to parameterize the Jastrow
            matrices. If None, all pairs are considered.
        occupied_orbitals: The occupied orbitals for the reference state. Defaults to
            the Hartree-Fock state.
        chunk_size: The number of two-body Hamiltonian terms to process at a time.
            If ``None``, all two-body terms are processed in one batch.
        method: The optimization method. See the documentation of
            `scipy.optimize.minimize`_ for possible values.
        callback: Callback function for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
        options: Options for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
        return_optimize_result: Whether to also return the `OptimizeResult`_ returned
            by `scipy.optimize.minimize`_.

    Returns:
        The optimized UCJ operator. If ``return_optimize_result`` is set to ``True``,
        the `OptimizeResult`_ returned by `scipy.optimize.minimize`_ is also returned.

    .. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    .. _OptimizeResult: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    _validate_ucj_op(initial_ucj_op)
    _validate_molecular_hamiltonian(hamiltonian, initial_ucj_op.norb)

    norb = initial_ucj_op.norb
    with_final_orbital_rotation = initial_ucj_op.final_orbital_rotation is not None
    triu_indices = list(itertools.combinations_with_replacement(range(norb), 2))
    mat_indices = list(itertools.product(range(norb), repeat=2))
    pairs_aa, pairs_ab, pairs_bb = (
        interaction_pairs if interaction_pairs is not None else (None, None, None)
    )
    validate_interaction_pairs(pairs_aa, ordered=False)
    validate_interaction_pairs(pairs_ab, ordered=True)
    validate_interaction_pairs(pairs_bb, ordered=False)
    pairs_aa = triu_indices if pairs_aa is None else pairs_aa
    pairs_ab = mat_indices if pairs_ab is None else pairs_ab
    pairs_bb = triu_indices if pairs_bb is None else pairs_bb
    interaction_pairs_key = (
        _interaction_pairs_key(pairs_aa),
        _interaction_pairs_key(pairs_ab),
        _interaction_pairs_key(pairs_bb),
    )
    occupied_orbitals_key = _occupied_orbitals_key(norb, nelec, occupied_orbitals)

    value_and_grad = _make_spin_unbalanced_objective(
        norb,
        interaction_pairs_key,
        with_final_orbital_rotation,
        occupied_orbitals_key,
        chunk_size,
    )
    result = _minimize_ucj_parameters(
        value_and_grad,
        initial_ucj_op.to_parameters(interaction_pairs),
        hamiltonian,
        method,
        callback,
        options,
    )
    optimized_ucj_op = UCJOpSpinUnbalanced.from_parameters(
        result.x,
        norb=initial_ucj_op.norb,
        n_reps=1,
        interaction_pairs=interaction_pairs,
        with_final_orbital_rotation=with_final_orbital_rotation,
    )

    if return_optimize_result:
        return optimized_ucj_op, result

    return optimized_ucj_op


def ucj_energy_spinless(
    ucj_op: UCJOpSpinless,
    hamiltonian: MolecularHamiltonianSpinless,
    nelec: int,
    *,
    occupied_orbitals: Sequence[int] | None = None,
    chunk_size: int | None = None,
) -> float:
    """Compute the UCJ energy for a spinless system
    using the fermionic backpropagation outlined in https://arxiv.org/abs/2607.21337.

    Args:
        hamiltonian: The Hamiltonian.
        ucj_op: The UCJ operator. Must have n_reps=1, with an optional final
            orbital rotation.
        nelec: The number of electrons.
        occupied_orbitals: The occupied orbitals for the reference state. Defaults to
            the Hartree-Fock state.
        chunk_size: The number of two-body Hamiltonian terms to process at a time.
            If ``None``, all two-body terms are processed in one batch. This is useful
            for large systems where the two-body tensor may not fit in GPU memory.

    Returns:
        The expectation value of the Hamiltonian with respect to the UCJ state.
    """
    _validate_ucj_op(ucj_op)
    _validate_molecular_hamiltonian(hamiltonian, ucj_op.norb)

    norb = ucj_op.norb
    orbital_rotation = ucj_op.orbital_rotations[0]
    final_orbital_rotation = ucj_op.final_orbital_rotation
    u = (
        orbital_rotation
        if final_orbital_rotation is None
        else final_orbital_rotation @ orbital_rotation
    )

    occupied = _occupied_orbitals_key_spinless(norb, nelec, occupied_orbitals)
    q = jnp.asarray(orbital_rotation.conj().T[:, occupied])
    h_bp, g_bp = _propagate_through_orbital_rotations(
        jnp.asarray(hamiltonian.one_body_tensor),
        jnp.asarray(hamiltonian.two_body_tensor),
        jnp.asarray(u),
    )
    jastrow_mat, jastrow_vec = _spinless_jastrow_phase(
        jnp.asarray(ucj_op.diag_coulomb_mats[0])
    )
    return float(
        _compute_energy_spinless(
            q,
            jnp.asarray(hamiltonian.constant),
            h_bp,
            g_bp,
            jastrow_mat,
            jastrow_vec,
            norb,
            chunk_size=chunk_size,
        )
    )


def ucj_energy_and_grad_spinless(
    ucj_op: UCJOpSpinless,
    hamiltonian: MolecularHamiltonianSpinless,
    nelec: int,
    interaction_pairs: list[tuple[int, int]] | None = None,
    *,
    occupied_orbitals: Sequence[int] | None = None,
    chunk_size: int | None = None,
) -> tuple[float, np.ndarray]:
    """Compute the UCJ energy and parameter gradient for a spinless system.

    The gradient is with respect to the flattened parameter vector returned by
    ``ucj_op.to_parameters(interaction_pairs=interaction_pairs)``.

    Args:
        hamiltonian: The Hamiltonian.
        ucj_op: The UCJ operator. Must have n_reps=1, with an optional final
            orbital rotation.
        nelec: The number of electrons.
        interaction_pairs: The interaction pairs used to parameterize the Jastrow
            matrix. If None, all pairs are considered.
        occupied_orbitals: The occupied orbitals for the reference state. Defaults to
            the Hartree-Fock state.
        chunk_size: The number of two-body Hamiltonian terms to process at a time.
            If ``None``, all two-body terms are processed in one batch.

    Returns:
        The energy and its gradient with respect to the UCJ parameter vector.
    """
    _validate_ucj_op(ucj_op)
    _validate_molecular_hamiltonian(hamiltonian, ucj_op.norb)

    norb = ucj_op.norb
    with_final_orbital_rotation = ucj_op.final_orbital_rotation is not None
    validate_interaction_pairs(interaction_pairs, ordered=False)
    interaction_pairs_resolved = (
        list(itertools.combinations_with_replacement(range(norb), 2))
        if interaction_pairs is None
        else interaction_pairs
    )
    interaction_pairs_key = _interaction_pairs_key(interaction_pairs_resolved)
    occupied_orbitals_key = _occupied_orbitals_key_spinless(
        norb, nelec, occupied_orbitals
    )

    value_and_grad = _make_spinless_objective(
        norb,
        interaction_pairs_key,
        with_final_orbital_rotation,
        occupied_orbitals_key,
        chunk_size,
    )
    return _evaluate_ucj_value_and_grad(
        value_and_grad,
        ucj_op.to_parameters(interaction_pairs=interaction_pairs),
        hamiltonian,
    )


def optimize_ucj_energy_spinless(
    initial_ucj_op: UCJOpSpinless,
    hamiltonian: MolecularHamiltonianSpinless,
    nelec: int,
    *,
    interaction_pairs: list[tuple[int, int]] | None = None,
    occupied_orbitals: Sequence[int] | None = None,
    chunk_size: int | None = None,
    method: str = "L-BFGS-B",
    callback=None,
    options: dict | None = None,
    return_optimize_result: bool = False,
) -> UCJOpSpinless | tuple[UCJOpSpinless, scipy.optimize.OptimizeResult]:
    """Optimize a spinless UCJ operator to minimize energy.

    Args:
        initial_ucj_op: The initial UCJ operator. Must have n_reps=1, with an
            optional final orbital rotation.
        hamiltonian: The Hamiltonian.
        nelec: The number of electrons.
        interaction_pairs: The interaction pairs used to parameterize the Jastrow
            matrix. If None, all pairs are considered.
        occupied_orbitals: The occupied orbitals for the reference state. Defaults to
            the Hartree-Fock state.
        chunk_size: The number of two-body Hamiltonian terms to process at a time.
            If ``None``, all two-body terms are processed in one batch.
        method: The optimization method. See the documentation of
            `scipy.optimize.minimize`_ for possible values.
        callback: Callback function for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
        options: Options for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
        return_optimize_result: Whether to also return the `OptimizeResult`_ returned
            by `scipy.optimize.minimize`_.

    Returns:
        The optimized UCJ operator. If ``return_optimize_result`` is set to ``True``,
        the `OptimizeResult`_ returned by `scipy.optimize.minimize`_ is also returned.

    .. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    .. _OptimizeResult: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    _validate_ucj_op(initial_ucj_op)
    _validate_molecular_hamiltonian(hamiltonian, initial_ucj_op.norb)

    norb = initial_ucj_op.norb
    with_final_orbital_rotation = initial_ucj_op.final_orbital_rotation is not None
    validate_interaction_pairs(interaction_pairs, ordered=False)
    interaction_pairs_resolved = (
        list(itertools.combinations_with_replacement(range(norb), 2))
        if interaction_pairs is None
        else interaction_pairs
    )
    interaction_pairs_key = _interaction_pairs_key(interaction_pairs_resolved)
    occupied_orbitals_key = _occupied_orbitals_key_spinless(
        norb, nelec, occupied_orbitals
    )

    value_and_grad = _make_spinless_objective(
        norb,
        interaction_pairs_key,
        with_final_orbital_rotation,
        occupied_orbitals_key,
        chunk_size,
    )
    result = _minimize_ucj_parameters(
        value_and_grad,
        initial_ucj_op.to_parameters(interaction_pairs=interaction_pairs),
        hamiltonian,
        method,
        callback,
        options,
    )
    optimized_ucj_op = UCJOpSpinless.from_parameters(
        result.x,
        norb=initial_ucj_op.norb,
        n_reps=1,
        interaction_pairs=interaction_pairs,
        with_final_orbital_rotation=with_final_orbital_rotation,
    )

    if return_optimize_result:
        return optimized_ucj_op, result

    return optimized_ucj_op


def _validate_ucj_op(ucj_op) -> None:
    """Check if the UCJ operator is compatible with fermionic backpropagation."""
    if ucj_op.n_reps != 1:
        raise NotImplementedError(
            "Fermionic backpropagation only supports UCJ operators with n_reps=1. "
            f"Got n_reps={ucj_op.n_reps}."
        )


def _validate_molecular_hamiltonian(
    hamiltonian: MolecularHamiltonian | MolecularHamiltonianSpinless, norb: int
) -> None:
    if hamiltonian.norb != norb:
        raise ValueError(
            "The Hamiltonian and UCJ operator should have the same number of "
            f"orbitals. Got {hamiltonian.norb} and {norb}."
        )


def _evaluate_ucj_value_and_grad(
    value_and_grad,
    params: np.ndarray,
    hamiltonian: MolecularHamiltonian | MolecularHamiltonianSpinless,
) -> tuple[float, np.ndarray]:
    """Evaluate a JAX UCJ objective and return NumPy-compatible outputs."""
    value, grad = value_and_grad(
        jnp.asarray(params),
        jnp.asarray(hamiltonian.one_body_tensor),
        jnp.asarray(hamiltonian.two_body_tensor),
        jnp.asarray(hamiltonian.constant),
    )
    return float(value), np.asarray(grad)


def _minimize_ucj_parameters(
    value_and_grad,
    initial_params: np.ndarray,
    hamiltonian: MolecularHamiltonian | MolecularHamiltonianSpinless,
    method: str,
    callback,
    options: dict | None,
) -> scipy.optimize.OptimizeResult:
    """Run SciPy minimization with a JAX value-and-gradient function."""
    one_body_tensor = jnp.asarray(hamiltonian.one_body_tensor)
    two_body_tensor = jnp.asarray(hamiltonian.two_body_tensor)
    constant = jnp.asarray(hamiltonian.constant)

    def scipy_func(x: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = value_and_grad(
            jnp.asarray(x),
            one_body_tensor,
            two_body_tensor,
            constant,
        )
        return float(value), np.asarray(grad)

    return scipy.optimize.minimize(
        scipy_func,
        initial_params,
        method=method,
        jac=True,
        callback=callback,
        options=options,
    )


def _interaction_pairs_key(
    interaction_pairs: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Functools requires arguments to be hashable."""
    return tuple((i, j) for i, j in interaction_pairs)


def _occupied_orbitals_key(
    norb: int,
    nelec: tuple[int, int],
    occupied_orbitals: tuple[Sequence[int], Sequence[int]] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Functools requires arguments to be hashable."""
    if occupied_orbitals is None:
        occupied_orbitals = (range(nelec[0]), range(nelec[1]))

    occupied_alpha = tuple(occupied_orbitals[0])
    occupied_beta = tuple(occupied_orbitals[1])
    if len(occupied_alpha) != nelec[0] or len(occupied_beta) != nelec[1]:
        raise ValueError(
            "occupied_orbitals should contain nelec[0] alpha orbitals and "
            "nelec[1] beta orbitals."
        )
    if any(orb < 0 or orb >= norb for orb in occupied_alpha):
        raise ValueError("Alpha occupied orbital indices are out of range.")
    if any(orb < 0 or orb >= norb for orb in occupied_beta):
        raise ValueError("Beta occupied orbital indices are out of range.")
    return occupied_alpha, occupied_beta


def _occupied_orbitals_key_spinless(
    norb: int,
    nelec: int,
    occupied_orbitals: Sequence[int] | None,
) -> tuple[int, ...]:
    """Resolve and validate spinless occupied reference orbital indices."""
    if occupied_orbitals is None:
        occupied_orbitals = range(nelec)

    occupied = tuple(occupied_orbitals)
    if len(occupied) != nelec:
        raise ValueError("occupied_orbitals should contain nelec orbitals.")
    if any(orb < 0 or orb >= norb for orb in occupied):
        raise ValueError("Occupied orbital indices are out of range.")
    return occupied


@functools.cache
def _make_spin_balanced_objective(
    norb: int,
    interaction_pairs_key: tuple[
        tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]
    ],
    with_final_orbital_rotation: bool,
    occupied_orbitals_key: tuple[tuple[int, ...], tuple[int, ...]],
    chunk_size: int | None,
):
    """Build a jitted value-and-gradient function for spin-balanced UCJ energy."""
    interaction_pairs = interaction_pairs_key
    occupied_alpha = jnp.asarray(occupied_orbitals_key[0])
    occupied_beta = jnp.asarray(occupied_orbitals_key[1])

    def energy(
        params: jax.Array,
        one_body_tensor: jax.Array,
        two_body_tensor: jax.Array,
        constant: jax.Array,
    ) -> jax.Array:
        index = 0
        n_orbital_rotation_params = norb**2
        orbital_rotation = unitary_from_parameters_jax(
            params[index : index + n_orbital_rotation_params], dim=norb
        )
        index += n_orbital_rotation_params

        diag_coulomb_mats = []
        for pairs in interaction_pairs:
            n_diag_coulomb_params = len(pairs)
            mat = real_symmetrics_from_parameters_jax(
                params[index : index + n_diag_coulomb_params],
                dim=norb,
                n_mats=1,
                triu_indices=pairs,
            )[0]
            index += n_diag_coulomb_params
            diag_coulomb_mats.append(mat)
        diag_coulomb_mats = jnp.stack(diag_coulomb_mats)

        final_orbital_rotation = None
        if with_final_orbital_rotation:
            final_orbital_rotation = unitary_from_parameters_jax(
                params[index:], dim=norb
            )

        u = (
            orbital_rotation
            if final_orbital_rotation is None
            else final_orbital_rotation @ orbital_rotation
        )
        h_bp, g_bp = _propagate_through_orbital_rotations(
            one_body_tensor, two_body_tensor, u
        )
        jastrow_mat, jastrow_vec = _spin_balanced_jastrow_phase(
            diag_coulomb_mats[0], diag_coulomb_mats[1], norb
        )
        rotated_reference = orbital_rotation.conj().T
        q_alpha = rotated_reference[:, occupied_alpha]
        q_beta = rotated_reference[:, occupied_beta]
        return _compute_energy_spin_balanced(
            q_alpha,
            q_beta,
            constant,
            h_bp,
            g_bp,
            jastrow_mat,
            jastrow_vec,
            norb,
            chunk_size=chunk_size,
        )

    return jax.jit(jax.value_and_grad(energy, argnums=0))


@functools.cache
def _make_spin_unbalanced_objective(
    norb: int,
    interaction_pairs_key: tuple[
        tuple[tuple[int, int], ...],
        tuple[tuple[int, int], ...],
        tuple[tuple[int, int], ...],
    ],
    with_final_orbital_rotation: bool,
    occupied_orbitals_key: tuple[tuple[int, ...], tuple[int, ...]],
    chunk_size: int | None,
):
    """Build a jitted value-and-gradient function for spin-unbalanced UCJ energy."""
    interaction_pairs = interaction_pairs_key
    occupied_alpha = jnp.asarray(occupied_orbitals_key[0])
    occupied_beta = jnp.asarray(occupied_orbitals_key[1])

    def energy(
        params: jax.Array,
        one_body_tensor: jax.Array,
        two_body_tensor: jax.Array,
        constant: jax.Array,
    ) -> jax.Array:
        pairs_aa, pairs_ab, pairs_bb = interaction_pairs
        index = 0

        orbital_rotations = []
        n_orbital_rotation_params = norb**2
        for _ in range(2):
            orbital_rotations.append(
                unitary_from_parameters_jax(
                    params[index : index + n_orbital_rotation_params], dim=norb
                )
            )
            index += n_orbital_rotation_params
        orbital_rotations = jnp.stack(orbital_rotations)

        mat_aa = real_symmetrics_from_parameters_jax(
            params[index : index + len(pairs_aa)],
            dim=norb,
            n_mats=1,
            triu_indices=pairs_aa,
        )[0]
        index += len(pairs_aa)
        mat_ab = jnp.zeros((norb, norb))
        if pairs_ab:
            rows, cols = zip(*pairs_ab)
            mat_ab = mat_ab.at[jnp.asarray(rows), jnp.asarray(cols)].set(
                params[index : index + len(pairs_ab)]
            )
        index += len(pairs_ab)
        mat_bb = real_symmetrics_from_parameters_jax(
            params[index : index + len(pairs_bb)],
            dim=norb,
            n_mats=1,
            triu_indices=pairs_bb,
        )[0]
        index += len(pairs_bb)
        diag_coulomb_mats = jnp.stack([mat_aa, mat_ab, mat_bb])

        final_orbital_rotation = None
        if with_final_orbital_rotation:
            final_orbital_rotation = []
            for _ in range(2):
                final_orbital_rotation.append(
                    unitary_from_parameters_jax(
                        params[index : index + n_orbital_rotation_params], dim=norb
                    )
                )
                index += n_orbital_rotation_params
            final_orbital_rotation = jnp.stack(final_orbital_rotation)
        orbital_rotation_alpha, orbital_rotation_beta = orbital_rotations
        u_alpha = (
            orbital_rotation_alpha
            if final_orbital_rotation is None
            else final_orbital_rotation[0] @ orbital_rotation_alpha
        )
        u_beta = (
            orbital_rotation_beta
            if final_orbital_rotation is None
            else final_orbital_rotation[1] @ orbital_rotation_beta
        )

        h_alpha, g_aa = _propagate_through_orbital_rotations(
            one_body_tensor, two_body_tensor, u_alpha
        )
        h_beta, g_bb = _propagate_through_orbital_rotations(
            one_body_tensor, two_body_tensor, u_beta
        )
        g_ab = _propagate_spin_sector_tensor(two_body_tensor, u_alpha, u_beta)
        g_ba = _propagate_spin_sector_tensor(two_body_tensor, u_beta, u_alpha)

        jastrow_mat, jastrow_vec = _spin_unbalanced_jastrow_phase(
            diag_coulomb_mats[0], diag_coulomb_mats[1], diag_coulomb_mats[2], norb
        )
        rotated_reference_alpha = orbital_rotation_alpha.conj().T
        rotated_reference_beta = orbital_rotation_beta.conj().T
        q_alpha = rotated_reference_alpha[:, occupied_alpha]
        q_beta = rotated_reference_beta[:, occupied_beta]
        return _compute_energy_spin_unbalanced(
            q_alpha,
            q_beta,
            constant,
            h_alpha,
            h_beta,
            g_aa,
            g_ab,
            g_ba,
            g_bb,
            jastrow_mat,
            jastrow_vec,
            norb,
            chunk_size=chunk_size,
        )

    return jax.jit(jax.value_and_grad(energy, argnums=0))


@functools.cache
def _make_spinless_objective(
    norb: int,
    interaction_pairs_key: tuple[tuple[int, int], ...],
    with_final_orbital_rotation: bool,
    occupied_orbitals_key: tuple[int, ...],
    chunk_size: int | None,
):
    """Build a jitted value-and-gradient function for spinless UCJ energy."""
    interaction_pairs = interaction_pairs_key
    occupied = jnp.asarray(occupied_orbitals_key)

    def energy(
        params: jax.Array,
        one_body_tensor: jax.Array,
        two_body_tensor: jax.Array,
        constant: jax.Array,
    ) -> jax.Array:
        index = 0
        n_orbital_rotation_params = norb**2
        orbital_rotation = unitary_from_parameters_jax(
            params[index : index + n_orbital_rotation_params], dim=norb
        )
        index += n_orbital_rotation_params

        diag_coulomb_mat = real_symmetrics_from_parameters_jax(
            params[index : index + len(interaction_pairs)],
            dim=norb,
            n_mats=1,
            triu_indices=interaction_pairs,
        )[0]
        index += len(interaction_pairs)

        final_orbital_rotation = None
        if with_final_orbital_rotation:
            final_orbital_rotation = unitary_from_parameters_jax(
                params[index:], dim=norb
            )

        u = (
            orbital_rotation
            if final_orbital_rotation is None
            else final_orbital_rotation @ orbital_rotation
        )
        h_bp, g_bp = _propagate_through_orbital_rotations(
            one_body_tensor, two_body_tensor, u
        )
        jastrow_mat, jastrow_vec = _spinless_jastrow_phase(diag_coulomb_mat)
        q = orbital_rotation.conj().T[:, occupied]
        return _compute_energy_spinless(
            q,
            constant,
            h_bp,
            g_bp,
            jastrow_mat,
            jastrow_vec,
            norb,
            chunk_size=chunk_size,
        )

    return jax.jit(jax.value_and_grad(energy, argnums=0))


def _propagate_through_orbital_rotations(h_pq, g_pqrs, u):
    r"""
    Propagate one- and two-body tensors through an orbital rotation
    using Lemma 1.

    .. math::
        H
        = E_0
        + \sum_{pq,\sigma} h_{pq} a^\dagger_{p\sigma} a_{q\sigma}
        + \frac12 \sum_{pqrs,\sigma\tau} g_{pqrs}
          a^\dagger_{p\sigma} a^\dagger_{r\tau} a_{s\tau} a_{q\sigma}.

    For the orbital rotation matrix :math:`u`, this computes

    .. math::
        \tilde{h}_{p' q'} =
        \sum_{pq} u^*_{p p'} h_{pq} u_{q q'}

    and

    .. math::
        \tilde{g}_{p' q' r' s'}
        = \sum_{p q r s}
            u^*_{p p'} u_{q q'} g_{p q r s} u^*_{r r'} u_{s s'}.

    Args:
        h_pq: The one-body tensor.
        g_pqrs: The two-body tensor.
        u: The orbital rotation matrix.
    Returns:
        The propagated one- and two-body tensors.
    """
    h_tilde = u.conj().T @ h_pq @ u
    g_tilde = jnp.einsum(
        "pi,qj,pqrs,rk,sl->ijkl",
        u.conj(),
        u,
        g_pqrs,
        u.conj(),
        u,
        optimize=True,
    )
    return h_tilde, g_tilde


def _propagate_spin_sector_tensor(g_pqrs, u_left, u_right):
    """Propagate two-body terms whose two one-body factors use different rotations."""
    return jnp.einsum(
        "pi,qj,pqrs,rk,sl->ijkl",
        u_left.conj(),
        u_left,
        g_pqrs,
        u_right.conj(),
        u_right,
        optimize=True,
    )


def _spin_balanced_jastrow_phase(same, diff, norb):
    r"""
    Convert spin-balanced Jastrow matrices to spin-orbital phase parameters.

    The Jastrow operator is given by

    .. math::
        J = \sum_{pq,\sigma\tau} j_{pq,\sigma\tau}
            \hat{n}_{p\sigma}\hat{n}_{q\tau}.

    This function rewrites the stored Jastrow generator as

    .. math::
        J(\mathbf{n}) = \mathbf{n}^T A \mathbf{n} + \mathbf{n}^T \ell,

    where ``jastrow_mat`` is :math:`A` and ``jastrow_vec`` is :math:`\ell`.
    The nested ``jastrow_phase`` helper below converts this representation into
    the scalar phases :math:`c` and vector phases :math:`\phi` of Lemma 2.

    Args:
        same: The spin-same Jastrow matrix.
        diff: The spin-different Jastrow matrix.
        norb: The number of orbitals.
    Returns:
        The spin-orbital Jastrow matrix and vector.
    """

    n_spin_orbitals = 2 * norb

    # the same-spin diagonal terms are idempotent and contribute to the linear term.
    same_offdiag = same - jnp.diag(jnp.diag(same))

    jastrow_mat = jnp.zeros((n_spin_orbitals, n_spin_orbitals))
    jastrow_mat = jastrow_mat.at[:norb, :norb].set(same_offdiag / 2)
    jastrow_mat = jastrow_mat.at[norb:, norb:].set(same_offdiag / 2)
    jastrow_mat = jastrow_mat.at[:norb, norb:].set(diff / 2)
    jastrow_mat = jastrow_mat.at[norb:, :norb].set(diff.T / 2)

    jastrow_vec = jnp.concatenate([jnp.diag(same) / 2, jnp.diag(same) / 2])
    return jastrow_mat, jastrow_vec


def _spin_unbalanced_jastrow_phase(same_aa, diff_ab, same_bb, norb):
    """Convert spin-unbalanced Jastrow matrices to spin-orbital phase parameters."""
    n_spin_orbitals = 2 * norb

    same_aa_offdiag = same_aa - jnp.diag(jnp.diag(same_aa))
    same_bb_offdiag = same_bb - jnp.diag(jnp.diag(same_bb))

    jastrow_mat = jnp.zeros((n_spin_orbitals, n_spin_orbitals))
    jastrow_mat = jastrow_mat.at[:norb, :norb].set(same_aa_offdiag / 2)
    jastrow_mat = jastrow_mat.at[norb:, norb:].set(same_bb_offdiag / 2)
    jastrow_mat = jastrow_mat.at[:norb, norb:].set(diff_ab / 2)
    jastrow_mat = jastrow_mat.at[norb:, :norb].set(diff_ab.T / 2)

    jastrow_vec = jnp.concatenate([jnp.diag(same_aa) / 2, jnp.diag(same_bb) / 2])
    return jastrow_mat, jastrow_vec


def _spinless_jastrow_phase(mat):
    """Convert a spinless Jastrow matrix to phase parameters."""
    mat_offdiag = mat - jnp.diag(jnp.diag(mat))
    return mat_offdiag / 2, jnp.diag(mat) / 2


def _transition_batch(phi, q):
    """Compute diagonal-phase Slater overlaps and transition densities."""
    q_conj = jnp.conj(q)
    n_occ = q.shape[1]
    norb = q.shape[0]
    d = jnp.exp(1j * phi)
    d_q = d[:, :, None] * q[None, :, :]
    overlap = jnp.einsum("pi,bpj->bij", q_conj, d_q)
    det = jnp.linalg.det(overlap)
    rhs = jnp.broadcast_to(q_conj.T, (phi.shape[0], n_occ, norb))
    transition_density = d_q @ jnp.linalg.solve(overlap, rhs)
    return det, transition_density


def _jastrow_phase(delta, jastrow_mat, jastrow_vec):
    """Return the Jastrow scalar and vector phase for occupation changes."""
    phi = -2.0 * (delta @ jastrow_mat.T)
    const = jnp.exp(
        -1j
        * (jnp.einsum("bi,ij,bj->b", delta, jastrow_mat, delta) + delta @ jastrow_vec)
    )
    return phi, const


def _spin_slice(spin: int, norb: int) -> slice:
    """Return the spin-orbital slice for alpha (0) or beta (1) orbitals."""
    offset = spin * norb
    return slice(offset, offset + norb)


@functools.cache
def _canonical_same_spin_two_body_indices(
    norb: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return canonical same-spin two-body index arrays.
    """
    pairs = list(itertools.combinations(range(norb), 2))
    n_terms = len(pairs) ** 2
    p = np.empty(n_terms, dtype=np.int64)
    q = np.empty(n_terms, dtype=np.int64)
    r = np.empty(n_terms, dtype=np.int64)
    s = np.empty(n_terms, dtype=np.int64)
    for index, ((p_, r_), (q_, s_)) in enumerate(itertools.product(pairs, repeat=2)):
        p[index] = p_
        q[index] = q_
        r[index] = r_
        s[index] = s_
    return p, q, r, s


def _chunked_term_sum(n_terms: int, chunk_size: int | None, func):
    """Sum per-term values over indices, optionally in fixed-size chunks."""
    if chunk_size is None or chunk_size >= n_terms:
        return jnp.sum(func(jnp.arange(n_terms)))
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    n_chunks = (n_terms + chunk_size - 1) // chunk_size
    padded_n_terms = n_chunks * chunk_size
    indices = jnp.arange(padded_n_terms)
    mask = indices < n_terms
    safe_indices = jnp.where(mask, indices, 0)
    index_chunks = safe_indices.reshape(n_chunks, chunk_size)
    mask_chunks = mask.reshape(n_chunks, chunk_size)

    def chunk_sum(args):
        chunk_indices, chunk_mask = args
        return jnp.sum(jnp.where(chunk_mask, func(chunk_indices), 0))

    return jnp.sum(jax.lax.map(jax.checkpoint(chunk_sum), (index_chunks, mask_chunks)))


def _one_body_spin_sector_energy(
    q_sector,
    q_other,
    h_sector,
    jastrow_mat,
    jastrow_vec,
    norb: int,
    spin: int,
):
    """Evaluate one-body terms for one spin sector of a spinful Hamiltonian."""
    n_spin_orbitals = 2 * norb
    p, q = jnp.meshgrid(jnp.arange(norb), jnp.arange(norb), indexing="ij")
    p = p.ravel()
    q = q.ravel()
    rows = jnp.arange(norb**2)
    offset = spin * norb

    delta = (
        jnp.zeros((norb**2, n_spin_orbitals))
        .at[rows, offset + p]
        .add(1)
        .at[rows, offset + q]
        .add(-1)
    )
    phi, const = _jastrow_phase(delta, jastrow_mat, jastrow_vec)
    sector_slice = _spin_slice(spin, norb)
    other_slice = _spin_slice(1 - spin, norb)
    det_sector, rho_sector = _transition_batch(phi[:, sector_slice], q_sector)
    det_other, _ = _transition_batch(phi[:, other_slice], q_other)

    return jnp.sum(
        h_sector[p, q] * const * det_sector * det_other * rho_sector[rows, q, p]
    )


def _same_spin_two_body_energy(
    q_sector,
    q_other,
    g,
    jastrow_mat,
    jastrow_vec,
    norb: int,
    spin: int,
    chunk_size: int | None,
):
    """Evaluate all canonical same-spin two-body terms."""
    n_terms = (norb * (norb - 1) // 2) ** 2
    if n_terms == 0:
        return 0.0
    p_all, q_all, r_all, s_all = (
        jnp.asarray(array)
        for array in _canonical_same_spin_two_body_indices(norb)
    )
    n_spin_orbitals = 2 * norb
    offset = spin * norb

    def term_chunk(indices):
        p = p_all[indices]
        q = q_all[indices]
        r = r_all[indices]
        s = s_all[indices]
        rows = jnp.arange(indices.shape[0])

        delta = (
            jnp.zeros((indices.shape[0], n_spin_orbitals))
            .at[rows, offset + p]
            .add(1)
            .at[rows, offset + r]
            .add(1)
            .at[rows, offset + s]
            .add(-1)
            .at[rows, offset + q]
            .add(-1)
        )
        phi, const = _jastrow_phase(delta, jastrow_mat, jastrow_vec)
        sector_slice = _spin_slice(spin, norb)
        other_slice = _spin_slice(1 - spin, norb)
        det_sector, rho_sector = _transition_batch(phi[:, sector_slice], q_sector)
        det_other, _ = _transition_batch(phi[:, other_slice], q_other)
        wick = (
            rho_sector[rows, q, p] * rho_sector[rows, s, r]
            - rho_sector[rows, s, p] * rho_sector[rows, q, r]
        )
        coeff = (
            g[p, q, r, s]
            - g[r, q, p, s]
            - g[p, s, r, q]
            + g[r, s, p, q]
        )
        return 0.5 * coeff * const * det_sector * det_other * wick

    return _chunked_term_sum(n_terms, chunk_size, term_chunk)


def _opposite_spin_two_body_energy(
    q_left,
    q_right,
    g,
    jastrow_mat,
    jastrow_vec,
    norb: int,
    left_spin: int,
    chunk_size: int | None,
):
    """Evaluate all opposite-spin two-body terms."""
    n_terms = norb**4
    g_flat = g.reshape(-1)
    n_spin_orbitals = 2 * norb
    left_offset = left_spin * norb
    right_spin = 1 - left_spin
    right_offset = right_spin * norb

    def term_chunk(indices):
        p, q, r, s = jnp.unravel_index(indices, (norb, norb, norb, norb))
        rows = jnp.arange(indices.shape[0])

        delta = (
            jnp.zeros((indices.shape[0], n_spin_orbitals))
            .at[rows, left_offset + p]
            .add(1)
            .at[rows, left_offset + q]
            .add(-1)
            .at[rows, right_offset + r]
            .add(1)
            .at[rows, right_offset + s]
            .add(-1)
        )
        phi, const = _jastrow_phase(delta, jastrow_mat, jastrow_vec)
        det_left, rho_left = _transition_batch(
            phi[:, _spin_slice(left_spin, norb)], q_left
        )
        det_right, rho_right = _transition_batch(
            phi[:, _spin_slice(right_spin, norb)], q_right
        )
        return (
            0.5
            * g_flat[indices]
            * const
            * det_left
            * det_right
            * rho_left[rows, q, p]
            * rho_right[rows, s, r]
        )

    return _chunked_term_sum(n_terms, chunk_size, term_chunk)


def _spinful_two_body_energy(
    q_alpha,
    q_beta,
    g_aa,
    g_ab,
    g_ba,
    g_bb,
    jastrow_mat,
    jastrow_vec,
    norb: int,
    chunk_size: int | None,
):
    """Evaluate all spin cases of the two-body Hamiltonian terms."""
    term_aa = _same_spin_two_body_energy(
        q_alpha, q_beta, g_aa, jastrow_mat, jastrow_vec, norb, 0, chunk_size
    )
    term_bb = _same_spin_two_body_energy(
        q_beta, q_alpha, g_bb, jastrow_mat, jastrow_vec, norb, 1, chunk_size
    )
    term_ab = _opposite_spin_two_body_energy(
        q_alpha, q_beta, g_ab, jastrow_mat, jastrow_vec, norb, 0, chunk_size
    )
    term_ba = _opposite_spin_two_body_energy(
        q_beta, q_alpha, g_ba, jastrow_mat, jastrow_vec, norb, 1, chunk_size
    )
    return term_aa + term_bb + term_ab + term_ba


def _compute_energy_spin_balanced(
    q_alpha,
    q_beta,
    constant,
    h_bp,
    g_bp,
    jastrow_mat,
    jastrow_vec,
    norb: int,
    *,
    chunk_size: int | None = None,
):
    r"""Compute the spin-balanced UCJ energy from backpropagated tensors.

    This implements the final energy calculation of fermionic backpropagation.

    .. math::
        E_{\mathrm{UCJ1}}
        := \langle \psi | H | \psi \rangle_{\mathrm{UCJ1}}
        = \langle Q | \tilde{H} | Q \rangle,

    where :math:`|Q\rangle := e^{-K}|\phi_0\rangle` and :math:`\tilde{H}` is
    the Hamiltonian after backpropagating through the orbital rotation and
    Jastrow operator.

    The propagation through the Jastrow operator gives

    .. math::
        e^{-iJ} a^\dagger_{p\sigma} a_{q\sigma} e^{iJ}
        = e^{i c^{[1]}_{pq\sigma}}
          a^\dagger_{p\sigma} a_{q\sigma}
          e^{i \phi^{[1]}_{pq\sigma} \cdot \hat{n}}.

    In the implementation, these phases are generated from the occupation-change
    vector :math:`\delta` using the equivalent representation

    .. math::
        J(\mathbf{n}) = \mathbf{n}^T A \mathbf{n} + \mathbf{n}^T \ell,

    giving

    .. math::
        \phi = -2 A^T \delta,
        \qquad
        e^{ic} = \exp[-i(\delta^T A \delta + \delta^T \ell)].

    Finally, the overlaps are evaluated using Lowdin's formula
    using Lemma 3.

    Args:
        q_alpha: Alpha block of :math:`Q`, shape ``(norb, nalpha)``.
        q_beta: Beta block of :math:`Q`, shape ``(norb, nbeta)``.
        constant: Constant term :math:`E_0` in the Hamiltonian.
        h_bp: Backpropagated one-body tensor :math:`\tilde{h}`.
        g_bp: Backpropagated two-body tensor :math:`\tilde{g}`.
        jastrow_mat: Spin-orbital Jastrow quadratic matrix :math:`A`.
        jastrow_vec: Spin-orbital Jastrow linear vector :math:`\ell`.
        norb: The number of spatial orbitals.
        chunk_size: Number of two-body tensor elements to process per chunk. If
            ``None``, all terms are processed in one batch.

    Returns:
        The real-valued spin-balanced UCJ energy.
    """
    energy_alpha_1 = _one_body_spin_sector_energy(
        q_alpha, q_beta, h_bp, jastrow_mat, jastrow_vec, norb, 0
    )
    energy_beta_1 = _one_body_spin_sector_energy(
        q_beta, q_alpha, h_bp, jastrow_mat, jastrow_vec, norb, 1
    )
    energy_2 = _spinful_two_body_energy(
        q_alpha,
        q_beta,
        g_bp,
        g_bp,
        g_bp,
        g_bp,
        jastrow_mat,
        jastrow_vec,
        norb,
        chunk_size,
    )

    return jnp.real(constant + energy_alpha_1 + energy_beta_1 + energy_2)


def _compute_energy_spin_unbalanced(
    q_alpha,
    q_beta,
    constant,
    h_alpha,
    h_beta,
    g_aa,
    g_ab,
    g_ba,
    g_bb,
    jastrow_mat,
    jastrow_vec,
    norb: int,
    *,
    chunk_size: int | None = None,
):
    r"""Compute the spin-unbalanced UCJ energy from backpropagated tensors.

    This is the spin-unbalanced analogue of :func:`_compute_energy_spin_balanced`.
    The alpha and beta orbital rotations are independent, so the backpropagated
    Hamiltonian has separate one-body tensors :math:`\tilde{h}^\alpha` and
    :math:`\tilde{h}^\beta`, and four two-body tensors
    :math:`\tilde{g}^{\alpha\alpha}`,
    :math:`\tilde{g}^{\alpha\beta}`, :math:`\tilde{g}^{\beta\alpha}`, and
    :math:`\tilde{g}^{\beta\beta}`.

    Jastrow propagation is still evaluated in the full spin-orbital occupation
    basis. For an excitation with occupation-change vector :math:`\delta`,

    .. math::
        \phi = -2 A^T \delta,
        \qquad
        e^{ic} = \exp[-i(\delta^T A \delta + \delta^T \ell)].

    Same-spin two-body terms use the Wick contraction

    .. math::
        \rho_{q p}\rho_{s r} - \rho_{s p}\rho_{q r},

    while opposite-spin terms factor into products of alpha and beta transition
    densities.
    """
    energy_alpha_1 = _one_body_spin_sector_energy(
        q_alpha, q_beta, h_alpha, jastrow_mat, jastrow_vec, norb, 0
    )
    energy_beta_1 = _one_body_spin_sector_energy(
        q_beta, q_alpha, h_beta, jastrow_mat, jastrow_vec, norb, 1
    )
    energy_2 = _spinful_two_body_energy(
        q_alpha,
        q_beta,
        g_aa,
        g_ab,
        g_ba,
        g_bb,
        jastrow_mat,
        jastrow_vec,
        norb,
        chunk_size,
    )

    return jnp.real(constant + energy_alpha_1 + energy_beta_1 + energy_2)


def _compute_energy_spinless(
    q,
    constant,
    h_bp,
    g_bp,
    jastrow_mat,
    jastrow_vec,
    norb: int,
    *,
    chunk_size: int | None = None,
):
    r"""Compute the spinless UCJ energy from backpropagated tensors.

    The input tensors are the Hamiltonian tensors after backpropagating through the
    orbital rotation. The remaining Jastrow propagation is evaluated from

    .. math::
        J(\mathbf{n}) = \mathbf{n}^T A \mathbf{n} + \mathbf{n}^T \ell.

    For each one- or two-body excitation, the occupation-change vector
    :math:`\delta` gives the diagonal phase

    .. math::
        \phi = -2 A^T \delta,
        \qquad
        e^{ic} = \exp[-i(\delta^T A \delta + \delta^T \ell)].

    Lowdin's formula gives the transition density :math:`\rho`, and the two-body
    contribution uses the spinless Wick contraction

    .. math::
        \rho_{q p}\rho_{s r} - \rho_{s p}\rho_{q r}.
    """
    p, q_ = jnp.meshgrid(jnp.arange(norb), jnp.arange(norb), indexing="ij")
    p = p.ravel()
    q_ = q_.ravel()
    rows = jnp.arange(norb**2)
    delta = jnp.zeros((norb**2, norb)).at[rows, p].add(1).at[rows, q_].add(-1)
    phi, const = _jastrow_phase(delta, jastrow_mat, jastrow_vec)
    det, rho = _transition_batch(phi, q)
    energy_1 = jnp.sum(h_bp[p, q_] * const * det * rho[rows, q_, p])

    n_terms = (norb * (norb - 1) // 2) ** 2
    p_all, q_all, r_all, s_all = (
        jnp.asarray(array)
        for array in _canonical_same_spin_two_body_indices(norb)
    )

    def two_body_chunk(indices):
        p = p_all[indices]
        q_ = q_all[indices]
        r = r_all[indices]
        s = s_all[indices]
        rows = jnp.arange(indices.shape[0])

        delta = (
            jnp.zeros((indices.shape[0], norb))
            .at[rows, p]
            .add(1)
            .at[rows, r]
            .add(1)
            .at[rows, s]
            .add(-1)
            .at[rows, q_]
            .add(-1)
        )
        phi, const = _jastrow_phase(delta, jastrow_mat, jastrow_vec)
        det, rho = _transition_batch(phi, q)
        wick = (
            rho[rows, q_, p] * rho[rows, s, r]
            - rho[rows, s, p] * rho[rows, q_, r]
        )
        coeff = (
            g_bp[p, q_, r, s]
            - g_bp[r, q_, p, s]
            - g_bp[p, s, r, q_]
            + g_bp[r, s, p, q_]
        )
        return 0.5 * coeff * const * det * wick

    energy_2 = (
        0.0 if n_terms == 0 else _chunked_term_sum(n_terms, chunk_size, two_body_chunk)
    )

    return jnp.real(constant + energy_1 + energy_2)
