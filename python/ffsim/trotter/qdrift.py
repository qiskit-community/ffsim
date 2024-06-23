# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import itertools

import numpy as np
import scipy.linalg

from ffsim.gates import apply_diag_coulomb_evolution, apply_num_op_sum_evolution
from ffsim.hamiltonians import DoubleFactorizedHamiltonian
from ffsim.states.wick import expectation_one_body_power, expectation_one_body_product


def simulate_qdrift_double_factorized(
    vec: np.ndarray,
    hamiltonian: DoubleFactorizedHamiltonian,
    time: float,
    *,
    norb: int,
    nelec: tuple[int, int],
    n_steps: int = 1,
    symmetric: bool = False,
    probabilities: str | np.ndarray = "norm",
    one_rdm: np.ndarray | None = None,
    n_samples: int = 1,
    seed=None,
) -> np.ndarray:
    """Double-factorized Hamiltonian simulation via qDRIFT.

    Args:
        vec: The state vector to evolve.
        hamiltonian: The Hamiltonian.
        time: The evolution time.
        nelec: The number of alpha and beta electrons.
        n_steps: The number of Trotter steps.
        probabilities: The sampling method to use, or else an explicit array of
            probabilities. If specifying a string, the following options are supported:
            - "norm": Sample each term with probability proportional to its
            spectral norm.
            - "uniform": Sample each term with uniform probability.
            - "optimal": Sample with probabilities optimized for a given initial state.
            The "optimal" method requires the one-body reduced density matrix of the
            initial state to be specified. It returns optimal probabilities whenever
            the initial state is completely characterized by this reduced density
            matrix, i.e., it is a Slater determinant.
        one_rdm: The one-body reduced density matrix of the initial state.
        n_samples: The number of qDRIFT trajectories to sample.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        A Numpy array representing the final state of the simulation. The shape of the
        array depends on the ``n_samples`` argument. If ``n_samples=1`` then it is
        just a state vector, a one-dimensional array. Otherwise, it is a two-dimensional
        array of shape ``(n_samples, dim)`` where ``dim`` is the dimension of the
        state vector.
    """
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}.")
    if n_samples < 1:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")

    initial_state = vec.copy()

    if n_steps == 0 or time == 0:
        if n_samples == 1:
            return initial_state
        return np.tile(initial_state, (n_samples, 1))

    if isinstance(probabilities, str):
        probabilities = qdrift_probabilities(
            hamiltonian, sampling_method=probabilities, nelec=nelec, one_rdm=one_rdm
        )
    if symmetric:
        # in symmetric qDRIFT the one-body term is treated as a constant
        probabilities[0] = 0
        probabilities /= sum(probabilities)

    rng = np.random.default_rng(seed)
    one_body_energies, one_body_basis_change = scipy.linalg.eigh(
        hamiltonian.one_body_tensor
    )
    step_time = time / n_steps

    results = np.empty((n_samples, initial_state.shape[0]), dtype=complex)
    for i in range(n_samples):
        # TODO cache trajectories
        vec = initial_state.copy()
        term_indices = rng.choice(
            len(probabilities), size=n_steps, replace=True, p=probabilities
        )
        if symmetric:
            vec = _simulate_qdrift_step_double_factorized_symmetric(
                vec,
                one_body_energies,
                one_body_basis_change,
                hamiltonian.diag_coulomb_mats,
                hamiltonian.orbital_rotations,
                step_time,
                z_representation=hamiltonian.z_representation,
                norb=norb,
                nelec=nelec,
                probabilities=probabilities,
                term_indices=term_indices,
            )
        else:
            vec = _simulate_qdrift_step_double_factorized(
                vec,
                one_body_energies,
                one_body_basis_change,
                hamiltonian.diag_coulomb_mats,
                hamiltonian.orbital_rotations,
                step_time,
                z_representation=hamiltonian.z_representation,
                norb=norb,
                nelec=nelec,
                probabilities=probabilities,
                term_indices=term_indices,
            )
        results[i] = vec

    if n_samples == 1:
        return results[0]
    return results


def _simulate_qdrift_step_double_factorized(
    vec: np.ndarray,
    one_body_energies: np.ndarray,
    one_body_basis_change: np.ndarray,
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    time: float,
    z_representation: bool,
    norb: int,
    nelec: tuple[int, int],
    probabilities: np.ndarray,
    term_indices: np.ndarray,
) -> np.ndarray:
    for term_index in term_indices:
        if term_index == 0:
            vec = apply_num_op_sum_evolution(
                vec,
                one_body_energies,
                time / probabilities[term_index],
                norb=norb,
                nelec=nelec,
                orbital_rotation=one_body_basis_change,
                copy=False,
            )
        else:
            vec = apply_diag_coulomb_evolution(
                vec,
                diag_coulomb_mats[term_index - 1],
                time / probabilities[term_index],
                norb=norb,
                nelec=nelec,
                orbital_rotation=orbital_rotations[term_index - 1],
                z_representation=z_representation,
                copy=False,
            )
    return vec


def _simulate_qdrift_step_double_factorized_symmetric(
    vec: np.ndarray,
    one_body_energies: np.ndarray,
    one_body_basis_change: np.ndarray,
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    time: float,
    z_representation: bool,
    norb: int,
    nelec: tuple[int, int],
    probabilities: np.ndarray,
    term_indices: np.ndarray,
) -> np.ndarray:
    # simulate the one-body term for half the time
    vec = apply_num_op_sum_evolution(
        vec,
        one_body_energies,
        0.5 * time,
        norb=norb,
        nelec=nelec,
        orbital_rotation=one_body_basis_change,
        copy=False,
    )
    # simulate the first sampled term
    term_index = term_indices[0]
    vec = apply_diag_coulomb_evolution(
        vec,
        diag_coulomb_mats[term_index - 1],
        time / probabilities[term_index],
        norb=norb,
        nelec=nelec,
        orbital_rotation=orbital_rotations[term_index - 1],
        z_representation=z_representation,
        copy=False,
    )
    # simulate the remaining sampled terms
    for term_index in term_indices[1:]:
        vec = apply_num_op_sum_evolution(
            vec,
            one_body_energies,
            time,
            norb=norb,
            nelec=nelec,
            orbital_rotation=one_body_basis_change,
            copy=False,
        )
        vec = apply_diag_coulomb_evolution(
            vec,
            diag_coulomb_mats[term_index - 1],
            time / probabilities[term_index],
            norb=norb,
            nelec=nelec,
            orbital_rotation=orbital_rotations[term_index - 1],
            z_representation=z_representation,
            copy=False,
        )
    # simulate the one-body term for half the time
    vec = apply_num_op_sum_evolution(
        vec,
        one_body_energies,
        0.5 * time,
        norb=norb,
        nelec=nelec,
        orbital_rotation=one_body_basis_change,
        copy=False,
    )

    return vec


def qdrift_probabilities(
    hamiltonian: DoubleFactorizedHamiltonian,
    sampling_method: str,
    *,
    nelec: tuple[int, int] | None = None,
    one_rdm: np.ndarray | None = None,
) -> np.ndarray:
    """Compute qDRIFT probabilities.

    Returns a list of probabilities for qDRIFT sampling. The first probability
    corresponds to the the one-body tensor of the Hamiltonian, and the rest
    correspond to the two-body terms in the same order as they are stored in, e.g.,
    ``hamiltonian.orbital_rotations``.

    Args:
        hamiltonian: The Hamiltonian
        sampling_method: The sampling method to use.
            The following options are supported:
            - "norm": Sample each term with probability proportional to its
            spectral norm.
            - "uniform": Sample each term with uniform probability.
            - "optimal": Sample with probabilities optimized for a given initial state.
            The "optimal" method requires the one-body reduced density matrix of the
            initial state to be specified.
            - "optimal-incoherent": Optimized probabilities for the incoherent qDRIFT
            channel.
        n_particles: The total number of particles.
        one_rdm: The one-body reduced density matrix of the initial state.

    Returns:
        The probabilities.
    """
    n_terms = 1 + len(hamiltonian.diag_coulomb_mats)

    if sampling_method == "norm":
        if nelec is None:
            raise ValueError(
                "The 'norm' sampling method requires nelec to be specified."
            )
        one_norms = np.zeros(n_terms)
        if np.all(np.linalg.matrix_rank(hamiltonian.diag_coulomb_mats) == 1):
            one_norms[0] = spectral_norm_one_body_tensor(
                hamiltonian.one_body_tensor, nelec=nelec
            )
        else:
            # when the diag coulomb mats have rank greater than one, only a loose
            # upper bound is used, so use a loose one for the one-body term too
            one_norms[0] = np.sum(
                np.abs(
                    scipy.linalg.eigh(hamiltonian.one_body_tensor, eigvals_only=True)
                )
            )
        for i, diag_coulomb_mat in enumerate(hamiltonian.diag_coulomb_mats):
            one_norms[i + 1] = spectral_norm_diag_coulomb(
                diag_coulomb_mat,
                z_representation=hamiltonian.z_representation,
                nelec=nelec,
            )
        return one_norms / np.sum(one_norms)

    elif sampling_method == "optimal":
        if one_rdm is None:
            raise ValueError(
                "The 'optimal' sampling method requires one_rdm to be " "specified."
            )
        variances = np.zeros(n_terms)
        one_body_tensor = scipy.linalg.block_diag(
            hamiltonian.one_body_tensor, hamiltonian.one_body_tensor
        )
        variances[0] = variance_one_body_tensor(one_rdm, one_body_tensor)
        for i in range(len(hamiltonian.diag_coulomb_mats)):
            variances[i + 1] = variance_diag_coulomb(
                one_rdm,
                hamiltonian.diag_coulomb_mats[i],
                orbital_rotation=hamiltonian.orbital_rotations[i],
                z_representation=hamiltonian.z_representation,
            )
        stds = np.sqrt(variances)
        return stds / np.sum(stds)

    elif sampling_method == "optimal-incoherent":
        if one_rdm is None:
            raise ValueError(
                "The 'optimal-incoherent' sampling method requires one_rdm to be "
                "specified."
            )
        variances = np.zeros(n_terms)
        one_body_tensor = scipy.linalg.block_diag(
            hamiltonian.one_body_tensor, hamiltonian.one_body_tensor
        )
        variances[0] = expectation_one_body_power(one_rdm, one_body_tensor, 2).real
        for i in range(len(hamiltonian.diag_coulomb_mats)):
            variances[i + 1] = expectation_squared_diag_coulomb(
                one_rdm,
                hamiltonian.diag_coulomb_mats[i],
                orbital_rotation=hamiltonian.orbital_rotations[i],
                z_representation=hamiltonian.z_representation,
            )
        stds = np.sqrt(variances)
        return stds / np.sum(stds)

    elif sampling_method == "uniform":
        return np.ones(n_terms) / n_terms

    raise ValueError(f"Unsupported sampling method: {sampling_method}.")


def spectral_norm_one_body_tensor(
    one_body_tensor: np.ndarray,
    *,
    nelec: tuple[int, int],
    z_representation: bool = False,
) -> float:
    """Compute an upper bound for the largest singular value of a one-body tensor.

    Args:
        one_body_tensor: The one-body tensor.
        nelec: The number of alpha and beta electrons.

    Returns:
        The upper bound.
    """
    eigs = scipy.linalg.eigh(one_body_tensor, eigvals_only=True)
    n_alpha, n_beta = nelec
    if z_representation:
        return 0.5 * max(
            abs(a + b)
            for a, b in itertools.product(
                [
                    sum(eigs[n_alpha:]) - sum(eigs[:n_alpha]),
                    sum(eigs[:-n_alpha]) - sum(eigs[-n_alpha:]),
                ],
                [
                    sum(eigs[n_beta:]) - sum(eigs[:n_beta]),
                    sum(eigs[:-n_beta]) - sum(eigs[-n_beta:]),
                ],
            )
        )
    return max(
        abs(a + b)
        for a, b in itertools.product(
            [sum(eigs[:n_alpha]), sum(eigs[-n_alpha:])],
            [sum(eigs[:n_beta]), sum(eigs[-n_beta:])],
        )
    )


def spectral_norm_diag_coulomb(
    diag_coulomb_mat: np.ndarray, nelec: tuple[int, int], z_representation: bool = False
) -> float:
    """Compute upper bound for the largest singular value of a diagonal Coulomb matrix.

    Args:
        diag_coulomb_mat: The diagonal Coulomb matrix.
        nelec: The number of alpha and beta electrons.
        z_representation: Whether the diagonal Coulomb matrix is in the
            Z representation.

    Returns:
        The upper bounds
    """
    # decompose the diag Coulomb mat as a sum of squared one-body operators
    one_body_tensors = one_body_square_decomposition(diag_coulomb_mat)

    # for a rank-1 diag Coulomb mat, we can compute the exact spectral norm
    if len(one_body_tensors) == 1:
        one_body_tensor = one_body_tensors[0]
        if z_representation:
            # TODO this abs is probably not necessary
            return abs(
                spectral_norm_one_body_tensor(
                    one_body_tensor, nelec=nelec, z_representation=True
                )
                ** 2
                - 0.25 * np.trace(diag_coulomb_mat)
            )
        return spectral_norm_one_body_tensor(one_body_tensor, nelec=nelec) ** 2

    # when the rank is greater than one, we only know how to return an upper bound
    if z_representation:
        return 0.5 * np.sum(np.abs(diag_coulomb_mat)) - 0.25 * np.sum(
            np.abs(np.diagonal(diag_coulomb_mat))
        )
    # TODO this is probably loose
    return 2 * np.sum(np.abs(diag_coulomb_mat))


def one_body_square_decomposition(
    diag_coulomb_mat: np.ndarray,
    orbital_rotation: np.ndarray | None = None,
    truncation_threshold: float = 1e-12,
) -> np.ndarray:
    """Decompose a two-body term as a sum of squared one-body operators.

    Args:
        diag_coulomb_mat: The diagonal Coulomb matrix.
        orbital_rotation: The orbital rotation.
        truncation_threshold: Eigenvalues of the diagonal Coulomb matrix with
            absolute value less than this value are truncated.
    """
    if orbital_rotation is None:
        orbital_rotation = np.eye(diag_coulomb_mat.shape[0])
    eigs, vecs = scipy.linalg.eigh(diag_coulomb_mat)
    index = np.abs(eigs) >= truncation_threshold
    eigs = eigs[index]
    vecs = vecs[:, index]
    return np.einsum(
        "t,it,ji,ki->tjk",
        np.emath.sqrt(0.5 * eigs),
        vecs,
        orbital_rotation,
        orbital_rotation.conj(),
    )


def variance_one_body_tensor(one_rdm: np.ndarray, one_body_tensor: np.ndarray) -> float:
    """Compute the variance of a one-body operator on a state given by its 1-RDM.

    Args:
        one_rdm: The one-body reduced density matrix of the state.
        one_body_tensor: The one-body tensor.

    Returns:
        The variance.
    """
    variance = (
        expectation_one_body_power(one_rdm, one_body_tensor, 2)
        - abs(expectation_one_body_power(one_rdm, one_body_tensor, 1)) ** 2
    ).real
    # value may be negative due to floating point error
    return max(0, variance)


def variance_diag_coulomb(
    one_rdm: np.ndarray,
    diag_coulomb_mat: np.ndarray,
    orbital_rotation: np.ndarray | None = None,
    z_representation: bool = False,
) -> float:
    """Compute the variance of a two-body operator on a state given by its 1-RDM.

    Args:
        one_rdm: The one-body reduced density matrix of the state.
        diag_coulomb_mat: The diagonal Coulomb matrix.
        orbital_rotation: The orbital rotation.
        z_representation: Whether the diagonal Coulomb matrix is in the
            Z representation.

    Returns:
        The variance.
    """
    if orbital_rotation is None:
        orbital_rotation = np.eye(diag_coulomb_mat.shape[0])
    one_body_ops = [
        scipy.linalg.block_diag(mat, mat)
        for mat in one_body_square_decomposition(diag_coulomb_mat, orbital_rotation)
    ]

    expectation: complex = 0
    expectation_squared: complex = 0

    for one_body_op in one_body_ops:
        expectation += expectation_one_body_power(one_rdm, one_body_op, 2)
        expectation_squared += expectation_one_body_power(one_rdm, one_body_op, 4)

    for one_body_1, one_body_2 in itertools.combinations(one_body_ops, 2):
        # this expectation is real so we can just double it instead of computing its
        # complex conjugate
        expectation_squared += 2 * expectation_one_body_product(
            one_rdm, [one_body_1, one_body_1, one_body_2, one_body_2]
        )

    if z_representation:
        one_body_correction = -0.5 * (
            np.einsum(
                "ij,pi,qi->pq",
                diag_coulomb_mat,
                orbital_rotation,
                orbital_rotation.conj(),
            )
            + np.einsum(
                "ij,pj,qj->pq",
                diag_coulomb_mat,
                orbital_rotation,
                orbital_rotation.conj(),
            )
        )
        one_body_correction = scipy.linalg.block_diag(
            one_body_correction, one_body_correction
        )
        expectation += expectation_one_body_power(one_rdm, one_body_correction, 1)
        expectation_squared += expectation_one_body_power(
            one_rdm, one_body_correction, 2
        )
        expectation_squared += expectation_one_body_product(
            one_rdm, [one_body_correction, one_body_op, one_body_op]
        )
        expectation_squared += expectation_one_body_product(
            one_rdm, [one_body_op, one_body_op, one_body_correction]
        )

    variance = (expectation_squared - abs(expectation) ** 2).real
    # value may be negative due to floating point error
    return max(0, variance)


def expectation_squared_diag_coulomb(
    one_rdm: np.ndarray,
    diag_coulomb_mat: np.ndarray,
    orbital_rotation: np.ndarray | None = None,
    z_representation: bool = False,
) -> float:
    """Compute the expectation squared of a diag Coulomb op.

    Args:
        one_rdm: The one-body reduced density matrix of the state.
        diag_coulomb_mat: The diagonal Coulomb matrix.
        orbital_rotation: The orbital rotation.
        z_representation: Whether the diagonal Coulomb matrix is in the
            Z representation.

    Returns:
        The expectation squared.
    """
    if orbital_rotation is None:
        orbital_rotation = np.eye(diag_coulomb_mat.shape[0])
    one_body_ops = [
        scipy.linalg.block_diag(mat, mat)
        for mat in one_body_square_decomposition(diag_coulomb_mat, orbital_rotation)
    ]

    expectation_squared: complex = 0

    for one_body_op in one_body_ops:
        expectation_squared += expectation_one_body_power(one_rdm, one_body_op, 4)

    for one_body_1, one_body_2 in itertools.combinations(one_body_ops, 2):
        # this expectation is real so we can just double it instead of computing its
        # complex conjugate
        expectation_squared += 2 * expectation_one_body_product(
            one_rdm, [one_body_1, one_body_1, one_body_2, one_body_2]
        )

    if z_representation:
        one_body_correction = -0.5 * (
            np.einsum(
                "ij,pi,qi->pq",
                diag_coulomb_mat,
                orbital_rotation,
                orbital_rotation.conj(),
            )
            + np.einsum(
                "ij,pj,qj->pq",
                diag_coulomb_mat,
                orbital_rotation,
                orbital_rotation.conj(),
            )
        )
        one_body_correction = scipy.linalg.block_diag(
            one_body_correction, one_body_correction
        )
        expectation_squared += expectation_one_body_power(
            one_rdm, one_body_correction, 2
        )
        expectation_squared += expectation_one_body_product(
            one_rdm, [one_body_correction, one_body_op, one_body_op]
        )
        expectation_squared += expectation_one_body_product(
            one_rdm, [one_body_op, one_body_op, one_body_correction]
        )

    return expectation_squared.real
