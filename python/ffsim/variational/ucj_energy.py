"""Implement the fermionic backpropagation algorithm for UCJ energy calculation."""

from __future__ import annotations

import functools
import itertools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

from ffsim.hamiltonians.molecular_hamiltonian import MolecularHamiltonian
from ffsim.linalg.util import (
    real_symmetrics_from_parameters_jax,
    unitary_from_parameters_jax,
)
from ffsim.variational.ucj_spin_balanced import UCJOpSpinBalanced
from ffsim.variational.ucj_spin_unbalanced import UCJOpSpinUnbalanced
from ffsim.variational.ucj_spinless import UCJOpSpinless
from ffsim.variational.util import validate_interaction_pairs

jax.config.update("jax_enable_x64", True)

# Conveience dispatchers
def ucj_energy(
    ucj_op: UCJOpSpinBalanced | UCJOpSpinUnbalanced | UCJOpSpinless,
    hamiltonian: MolecularHamiltonian,
    nelec: int | tuple[int, int],
    interaction_pairs=None,
    *,
    occupied_orbitals=None,
    chunk_size: int | None = None,
) -> float:
    """Convenience dispatcher for UCJ energy calculation."""

    if isinstance(ucj_op, UCJOpSpinBalanced):
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
    if isinstance(ucj_op, UCJOpSpinUnbalanced):
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
    if isinstance(ucj_op, UCJOpSpinless):
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

def optimize_ucj_energy(
    initial_ucj_op: UCJOpSpinBalanced | UCJOpSpinUnbalanced | UCJOpSpinless,
    hamiltonian: MolecularHamiltonian,
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

    if isinstance(initial_ucj_op, UCJOpSpinBalanced):
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

    if isinstance(initial_ucj_op, UCJOpSpinUnbalanced):
        if not isinstance(nelec, tuple):
            raise TypeError("nelec must be a tuple for a spin-unbalanced UCJ operator.")
        raise NotImplementedError(
            "Spin-unbalanced UCJ energy optimization is not implemented yet."
        )

    if isinstance(initial_ucj_op, UCJOpSpinless):
        if not isinstance(nelec, int):
            raise TypeError("nelec must be an int for a spinless UCJ operator.")
        raise NotImplementedError(
            "Spinless UCJ energy optimization is not implemented yet."
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
    validate_interaction_pairs(pairs_ab, ordered=True)

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

    q_alpha, q_beta = _reference_orbital_matrices(
        orbital_rotation, nelec, occupied_orbitals
    )

    h_pq, g_pqrs = _propagate_through_orbital_rotations(
        jnp.asarray(hamiltonian.one_body_tensor),
        jnp.asarray(hamiltonian.two_body_tensor),
        jnp.asarray(u),
    )

    jastrow_mat, jastrow_vec = _jastrow_phase_parameters_spin_balanced(
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
    pairs_aa, pairs_ab = _resolve_interaction_pairs(norb, interaction_pairs)
    interaction_pairs_key = (
        _interaction_pairs_key(pairs_aa),
        _interaction_pairs_key(pairs_ab),
    )
    occupied_orbitals_key = _occupied_orbitals_key(norb, nelec, occupied_orbitals)

    value_and_grad = _make_ucj_energy_and_grad_spin_balanced(
        norb,
        interaction_pairs_key,
        with_final_orbital_rotation,
        occupied_orbitals_key,
        chunk_size,
    )
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

    result = scipy.optimize.minimize(
        scipy_func,
        initial_ucj_op.to_parameters(interaction_pairs=interaction_pairs),
        method=method,
        jac=True,
        callback=callback,
        options=options,
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
    pairs_aa, pairs_ab, pairs_bb = (
        interaction_pairs if interaction_pairs is not None else (None, None, None)
    )
    validate_interaction_pairs(pairs_aa, ordered=False)
    validate_interaction_pairs(pairs_ab, ordered=True)
    validate_interaction_pairs(pairs_bb, ordered=False)
    raise NotImplementedError("Spin-unbalanced UCJ energy is not implemented yet.")

def ucj_energy_spinless(
    ucj_op: UCJOpSpinless,
    hamiltonian: MolecularHamiltonian,
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

    Returns:
        The expectation value of the Hamiltonian with respect to the UCJ state.
    """
    _validate_ucj_op(ucj_op)
    raise NotImplementedError("Spinless UCJ energy is not implemented yet.")

def _validate_ucj_op(ucj_op) -> None:
    """Check if the UCJ operator is compatible with fermionic backpropagation."""
    if ucj_op.n_reps != 1:
        raise NotImplementedError(
            "Fermionic backpropagation only supports UCJ operators with n_reps=1. "
            f"Got n_reps={ucj_op.n_reps}."
        )

def _validate_molecular_hamiltonian(
    hamiltonian: MolecularHamiltonian, norb: int
) -> None:
    if hamiltonian.norb != norb:
        raise ValueError(
            "The Hamiltonian and UCJ operator should have the same number of "
            f"orbitals. Got {hamiltonian.norb} and {norb}."
        )

def _resolve_interaction_pairs(
    norb: int,
    interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]
    | None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Resolve the interaction pairs for a spin-balanced UCJ operator.
    
    If interaction pairs are not provided, we use all pairs of orbitals, enumerated 
    using `itertools.combinations_with_replacement`.
    """
    triu_indices = list(itertools.combinations_with_replacement(range(norb), 2))
    pairs_aa, pairs_ab = (
        interaction_pairs if interaction_pairs is not None else (None, None)
    )
    validate_interaction_pairs(pairs_aa, ordered=False)
    validate_interaction_pairs(pairs_ab, ordered=True)
    return (
        triu_indices if pairs_aa is None else pairs_aa,
        triu_indices if pairs_ab is None else pairs_ab,
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
        raise ValueError("occupied_orbitals should contain nelec[0] alpha orbitals and nelec[1] beta orbitals.")
    if any(orb < 0 or orb >= norb for orb in occupied_alpha):
        raise ValueError("Alpha occupied orbital indices are out of range.")
    if any(orb < 0 or orb >= norb for orb in occupied_beta):
        raise ValueError("Beta occupied orbital indices are out of range.")
    return occupied_alpha, occupied_beta

def _reference_orbital_matrices(
    orbital_rotation: np.ndarray,
    nelec: tuple[int, int],
    occupied_orbitals: tuple[Sequence[int], Sequence[int]] | None,
) -> tuple[jax.Array, jax.Array]:
    r"""
    Construct the Slater determinant orbital matrices used in Lemma 3.

    The final energy is evaluated with the Slater determinant

    .. math::
        |Q\rangle := e^{-K} |\phi_0\rangle.

    This function returns the alpha and beta blocks of the matrix :math:`Q`.
    If :math:`o_\alpha` and :math:`o_\beta` are the occupied orbital index
    tuples, then

    .. math::
        (Q_\sigma)_{p i} = (u^\dagger)_{p, o_{\sigma, i}},
        \qquad \sigma \in \{\alpha, \beta\},

    where :math:`u` is ``orbital_rotation``.
    """
    norb = orbital_rotation.shape[0]
    occupied_alpha, occupied_beta = _occupied_orbitals_key(
        norb, nelec, occupied_orbitals
    )

    rotated_reference = orbital_rotation.conj().T
    return (
        jnp.asarray(rotated_reference[:, occupied_alpha]),
        jnp.asarray(rotated_reference[:, occupied_beta]),
    )

def _ucj_arrays_from_parameters_spin_balanced_jax(
    params,
    norb: int,
    interaction_pairs: tuple[Sequence[tuple[int, int]], Sequence[tuple[int, int]]],
    with_final_orbital_rotation: bool,
):
    """
    Rebuild jax arrays from the UCJ parameters.
    """
    pairs_aa, pairs_ab = interaction_pairs
    index = 0

    n_orbital_rotation_params = norb**2
    orbital_rotation = unitary_from_parameters_jax(
        params[index : index + n_orbital_rotation_params], dim=norb
    )
    index += n_orbital_rotation_params

    diag_coulomb_mats = []
    for pairs in (pairs_aa, pairs_ab):
        n_diag_coulomb_params = len(pairs)
        mat = real_symmetrics_from_parameters_jax(
            params[index : index + n_diag_coulomb_params],
            dim=norb,
            n_mats=1,
            triu_indices=pairs,
        )[0]
        index += n_diag_coulomb_params
        diag_coulomb_mats.append(mat)

    final_orbital_rotation = None
    if with_final_orbital_rotation:
        final_orbital_rotation = unitary_from_parameters_jax(params[index:], dim=norb)

    return orbital_rotation, jnp.stack(diag_coulomb_mats), final_orbital_rotation

@functools.cache
def _make_ucj_energy_and_grad_spin_balanced(
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
        orbital_rotation, diag_coulomb_mats, final_orbital_rotation = (
            _ucj_arrays_from_parameters_spin_balanced_jax(
                params,
                norb,
                interaction_pairs,
                with_final_orbital_rotation,
            )
        )
        u = (
            orbital_rotation
            if final_orbital_rotation is None
            else final_orbital_rotation @ orbital_rotation
        )
        h_bp, g_bp = _propagate_through_orbital_rotations(
            one_body_tensor, two_body_tensor, u
        )
        jastrow_mat, jastrow_vec = _jastrow_phase_parameters_spin_balanced(
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

def _propagate_through_orbital_rotations(h_pq, g_pqrs, u):
    r"""
    Propagate one- and two-body tensors through an orbital rotation
    (this is Lemma 1 in the paper).

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

def _jastrow_phase_parameters_spin_balanced(same, diff, norb):
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
    (which is Lemma 3 in the paper).

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
    n_spin_orbitals = 2 * norb

    def transition_batch(phi, q):
        r"""Compute diagonal-phase Slater overlaps and transition densities.

        For a Slater determinant with occupied orbital matrix :math:`Q` and 
        diagonal phase matrix

        .. math::
            D_\phi := \operatorname{diag}(e^{i\phi_1}, \ldots, e^{i\phi_N}),

        we get

        .. math::
            S := Q^\dagger D_\phi Q

        and return the determinant :math:`\det(S)` and

        .. math::
            \rho = D_\phi Q S^{-1} Q^\dagger.

        Lemma 3 then gives

        .. math::
            \langle Q | a^\dagger_p a_q e^{i\phi\cdot\hat{n}} | Q \rangle
            = \det(S) \rho_{q p}.
        """
        q_conj = jnp.conj(q)
        n_occ = q.shape[1]
        d = jnp.exp(1j * phi)
        d_q = d[:, :, None] * q[None, :, :]
        overlap = jnp.einsum("pi,bpj->bij", q_conj, d_q)
        det = jnp.linalg.det(overlap)
        rhs = jnp.broadcast_to(q_conj.T, (phi.shape[0], n_occ, norb))
        transition_density = d_q @ jnp.linalg.solve(overlap, rhs)
        return det, transition_density

    def jastrow_phase(delta):
        r"""Return the Jastrow phase induced by an occupation change.

        ``delta`` has one row per excitation and one column per spin orbital.
        Entries are ``+1`` for created orbitals and ``-1`` for annihilated
        orbitals. For each row :math:`\delta`, this returns the vector
        phase :math:`\phi` and scalar phase :math:`e^{ic}` such that

        .. math::
            e^{-iJ} X_\delta e^{iJ}
            = e^{ic} X_\delta e^{i\phi\cdot\hat{n}}.
        """
        phi = -2.0 * (delta @ jastrow_mat.T)
        const = jnp.exp(
            -1j
            * (
                jnp.einsum("bi,ij,bj->b", delta, jastrow_mat, delta)
                + delta @ jastrow_vec
            )
        )
        return phi, const

    p1, q1 = jnp.meshgrid(jnp.arange(norb), jnp.arange(norb), indexing="ij")
    p1 = p1.ravel()
    q1 = q1.ravel()
    rows1 = jnp.arange(norb**2)

    delta_alpha_1 = (
        jnp.zeros((norb**2, n_spin_orbitals))
        .at[rows1, p1]
        .add(1)
        .at[rows1, q1]
        .add(-1)
    )
    phi_alpha_1, const_alpha_1 = jastrow_phase(delta_alpha_1)
    det_alpha_1, rho_alpha_1 = transition_batch(phi_alpha_1[:, :norb], q_alpha)
    det_beta_1, _ = transition_batch(phi_alpha_1[:, norb:], q_beta)
    energy_alpha_1 = jnp.sum(
        h_bp[p1, q1]
        * const_alpha_1
        * det_alpha_1
        * det_beta_1
        * rho_alpha_1[rows1, q1, p1]
    )

    delta_beta_1 = (
        jnp.zeros((norb**2, n_spin_orbitals))
        .at[rows1, norb + p1]
        .add(1)
        .at[rows1, norb + q1]
        .add(-1)
    )
    phi_beta_1, const_beta_1 = jastrow_phase(delta_beta_1)
    det_alpha_1, _ = transition_batch(phi_beta_1[:, :norb], q_alpha)
    det_beta_1, rho_beta_1 = transition_batch(phi_beta_1[:, norb:], q_beta)
    energy_beta_1 = jnp.sum(
        h_bp[p1, q1]
        * const_beta_1
        * det_alpha_1
        * det_beta_1
        * rho_beta_1[rows1, q1, p1]
    )

    n4 = norb**4
    g_flat = g_bp.reshape(-1)

    def two_body_chunk(indices):
        r"""Evaluate a batch of two-body Hamiltonian tensor elements.

        Each flattened index selects one :math:`(p, q, r, s)` tensor element.
        The spin-balanced Hamiltonian contributes four spin cases:
        alpha-alpha, beta-beta, alpha-beta, and beta-alpha. Same-spin terms use
        the two-orbital Lowdin formula to evaluate the Wick contraction

        .. math::
            \rho_{q p}\rho_{s r} - \rho_{s p}\rho_{q r},

        while opposite-spin terms factor into alpha and beta one-body transition
        densities because the spin sectors are independent Slater determinants.
        """
        p, q, r, s = jnp.unravel_index(indices, (norb, norb, norb, norb))
        rows = jnp.arange(indices.shape[0])
        g = g_flat[indices]

        # alpha-alpha: a^\dagger_{p alpha} a^\dagger_{r alpha}
        #              a_{s alpha} a_{q alpha}
        delta_aa = (
            jnp.zeros((indices.shape[0], n_spin_orbitals))
            .at[rows, p]
            .add(1)
            .at[rows, r]
            .add(1)
            .at[rows, s]
            .add(-1)
            .at[rows, q]
            .add(-1)
        )
        phi_aa, const_aa = jastrow_phase(delta_aa)
        det_alpha_aa, rho_alpha_aa = transition_batch(phi_aa[:, :norb], q_alpha)
        det_beta_aa, _ = transition_batch(phi_aa[:, norb:], q_beta)
        wick_aa = (
            rho_alpha_aa[rows, q, p] * rho_alpha_aa[rows, s, r]
            - rho_alpha_aa[rows, s, p] * rho_alpha_aa[rows, q, r]
        )
        term_aa = 0.5 * g * const_aa * det_alpha_aa * det_beta_aa * wick_aa

        # beta-beta: same contraction as alpha-alpha, shifted into the beta block.
        delta_bb = (
            jnp.zeros((indices.shape[0], n_spin_orbitals))
            .at[rows, norb + p]
            .add(1)
            .at[rows, norb + r]
            .add(1)
            .at[rows, norb + s]
            .add(-1)
            .at[rows, norb + q]
            .add(-1)
        )
        phi_bb, const_bb = jastrow_phase(delta_bb)
        det_alpha_bb, _ = transition_batch(phi_bb[:, :norb], q_alpha)
        det_beta_bb, rho_beta_bb = transition_batch(phi_bb[:, norb:], q_beta)
        wick_bb = (
            rho_beta_bb[rows, q, p] * rho_beta_bb[rows, s, r]
            - rho_beta_bb[rows, s, p] * rho_beta_bb[rows, q, r]
        )
        term_bb = 0.5 * g * const_bb * det_alpha_bb * det_beta_bb * wick_bb

        # alpha-beta: alpha excitation a^\dagger_p a_q and beta excitation
        # a^\dagger_r a_s. The spin sectors factor after the Jastrow phase is
        # converted to diagonal one-body phases.
        delta_ab = (
            jnp.zeros((indices.shape[0], n_spin_orbitals))
            .at[rows, p]
            .add(1)
            .at[rows, q]
            .add(-1)
            .at[rows, norb + r]
            .add(1)
            .at[rows, norb + s]
            .add(-1)
        )
        phi_ab, const_ab = jastrow_phase(delta_ab)
        det_alpha_ab, rho_alpha_ab = transition_batch(phi_ab[:, :norb], q_alpha)
        det_beta_ab, rho_beta_ab = transition_batch(phi_ab[:, norb:], q_beta)
        term_ab = (
            0.5
            * g
            * const_ab
            * det_alpha_ab
            * det_beta_ab
            * rho_alpha_ab[rows, q, p]
            * rho_beta_ab[rows, s, r]
        )

        # beta-alpha: beta excitation a^\dagger_p a_q and alpha excitation
        # a^\dagger_r a_s.
        delta_ba = (
            jnp.zeros((indices.shape[0], n_spin_orbitals))
            .at[rows, norb + p]
            .add(1)
            .at[rows, norb + q]
            .add(-1)
            .at[rows, r]
            .add(1)
            .at[rows, s]
            .add(-1)
        )
        phi_ba, const_ba = jastrow_phase(delta_ba)
        det_alpha_ba, rho_alpha_ba = transition_batch(phi_ba[:, :norb], q_alpha)
        det_beta_ba, rho_beta_ba = transition_batch(phi_ba[:, norb:], q_beta)
        term_ba = (
            0.5
            * g
            * const_ba
            * det_alpha_ba
            * det_beta_ba
            * rho_beta_ba[rows, q, p]
            * rho_alpha_ba[rows, s, r]
        )

        return jnp.sum(term_aa + term_bb + term_ab + term_ba)

    # Chunking keeps the peak memory under control on large systems.
    if chunk_size is None or chunk_size >= n4:
        energy_2 = two_body_chunk(jnp.arange(n4))
    else:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if n4 % chunk_size:
            raise ValueError(
                f"chunk_size ({chunk_size}) must evenly divide norb**4 ({n4})."
            )
        index_chunks = jnp.arange(n4).reshape(-1, chunk_size)
        energy_2 = jnp.sum(jax.lax.map(jax.checkpoint(two_body_chunk), index_chunks))

    return jnp.real(constant + energy_alpha_1 + energy_beta_1 + energy_2)
