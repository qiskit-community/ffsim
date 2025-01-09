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
from collections import defaultdict

import numpy as np

from ffsim import hamiltonians, operators, variational
from ffsim.variational.util import validate_interaction_pairs


def random_state_vector(dim: int, *, seed=None, dtype=complex) -> np.ndarray:
    """Return a random state vector sampled from the uniform distribution.

    Args:
        dim: The dimension of the state vector.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.
        dtype: The data type to use for the result.

    Returns:
        The sampled state vector.

    Raises:
        ValueError: Dimension must be at least one.
    """
    if dim < 1:
        raise ValueError("Dimension must be at least one.")

    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(dtype, copy=False)
    if np.issubdtype(dtype, np.complexfloating):
        vec += 1j * rng.standard_normal(dim).astype(dtype, copy=False)
    vec /= np.linalg.norm(vec)
    return vec


def random_density_matrix(dim: int, *, seed=None, dtype=complex) -> np.ndarray:
    """Returns a random density matrix distributed with Hilbert-Schmidt measure.

    A density matrix is positive semi-definite and has trace equal to one.

    Args:
        dim: The width and height of the matrix.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled density matrix.

    Raises:
        ValueError: Dimension must be at least one.

    References:
        - `arXiv:0909.5094`_

    .. _arXiv:0909.5094: https://arxiv.org/abs/0909.5094
    """
    if dim < 1:
        raise ValueError("Dimension must be at least one.")

    rng = np.random.default_rng(seed)

    mat = rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    if np.issubdtype(dtype, np.complexfloating):
        mat += 1j * rng.standard_normal((dim, dim))
    mat @= mat.T.conj()
    mat /= np.trace(mat)
    return mat


def random_unitary(dim: int, *, seed=None, dtype=complex) -> np.ndarray:
    """Return a random unitary matrix distributed with Haar measure.

    Args:
        dim: The width and height of the matrix.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.
        dtype: The data type to use for the result.

    Returns:
        The sampled unitary matrix.

    References:
        - `arXiv:math-ph/0609050`_

    .. _arXiv:math-ph/0609050: https://arxiv.org/abs/math-ph/0609050
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    z += 1j * rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    return q * (d / np.abs(d))


def random_orthogonal(dim: int, seed=None, dtype=float) -> np.ndarray:
    """Return a random orthogonal matrix distributed with Haar measure.

    Args:
        dim: The width and height of the matrix.
        seed: The pseudorandom number generator or seed. Should be an
            instance of `np.random.Generator` or else a valid input to
            `np.random.default_rng`.

    Returns:
        The sampled orthogonal matrix.

    References:
        - `arXiv:math-ph/0609050`_

    .. _arXiv:math-ph/0609050: https://arxiv.org/abs/math-ph/0609050
    """
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    q, r = np.linalg.qr(m)
    d = np.diagonal(r)
    return q * (d / np.abs(d))


def random_special_orthogonal(dim: int, seed=None, dtype=float) -> np.ndarray:
    """Return a random special orthogonal matrix distributed with Haar measure.

    Args:
        dim: The width and height of the matrix.
        seed: The pseudorandom number generator or seed. Should be an
            instance of `np.random.Generator` or else a valid input to
            `np.random.default_rng`.

    Returns:
        The sampled special orthogonal matrix.
    """
    mat = random_orthogonal(dim, seed=seed, dtype=dtype)
    if np.linalg.det(mat) < 0:
        mat[0] *= -1
    return mat


def random_hermitian(dim: int, *, seed=None, dtype=complex) -> np.ndarray:
    """Return a random Hermitian matrix.

    Args:
        dim: The width and height of the matrix.
        rank: The rank of the matrix. If `None`, the maximum rank is used.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.
        dtype: The data type to use for the result.

    Returns:
        The sampled Hermitian matrix.
    """
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    mat += 1j * rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    return mat + mat.T.conj()


def random_real_symmetric_matrix(
    dim: int, *, rank: int | None = None, seed=None, dtype=float
) -> np.ndarray:
    """Return a random real symmetric matrix.

    Args:
        dim: The width and height of the matrix.
        rank: The rank of the matrix. If `None`, the maximum rank is used.
        seed: The pseudorandom number generator or seed. Should be an
            instance of `np.random.Generator` or else a valid input to
            `np.random.default_rng`.

    Returns:
        The sampled real symmetric matrix.
    """
    rng = np.random.default_rng(seed)
    if rank is None:
        rank = dim
    mat = rng.standard_normal((dim, rank)).astype(dtype, copy=False)
    return mat @ mat.T


def random_antihermitian(dim: int, *, seed=None, dtype=complex) -> np.ndarray:
    """Return a random anti-Hermitian matrix.

    Args:
        dim: The width and height of the matrix.
        rank: The rank of the matrix. If `None`, the maximum rank is used.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.
        dtype: The data type to use for the result.

    Returns:
        The sampled anti-Hermitian matrix.
    """
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    mat += 1j * rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    return mat - mat.T.conj()


def random_two_body_tensor(
    dim: int, *, rank: int | None = None, seed=None, dtype=complex
) -> np.ndarray:
    """Sample a random two-body tensor.

    Args:
        dim: The dimension of the tensor. The shape of the returned tensor will be
            (dim, dim, dim, dim).
        rank: Rank of the sampled tensor. The default behavior is to use
            the maximum rank, which is `norb * (norb + 1) // 2`.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.
        dtype: The data type to use for the result.

    Returns:
        The sampled two-body tensor.
    """
    rng = np.random.default_rng(seed)
    if rank is None:
        rank = dim * (dim + 1) // 2
    cholesky_vecs = rng.standard_normal((rank, dim, dim)).astype(dtype, copy=False)
    cholesky_vecs += cholesky_vecs.transpose((0, 2, 1))
    two_body_tensor = np.einsum("ipr,iqs->prqs", cholesky_vecs, cholesky_vecs)
    if np.issubdtype(dtype, np.complexfloating):
        orbital_rotation = random_unitary(dim, seed=rng)
        two_body_tensor = np.einsum(
            "abcd,aA,bB,cC,dD->ABCD",
            two_body_tensor,
            orbital_rotation,
            orbital_rotation.conj(),
            orbital_rotation,
            orbital_rotation.conj(),
            optimize=True,
        )
    return two_body_tensor


def random_t2_amplitudes(
    norb: int, nocc: int, *, seed=None, dtype=complex
) -> np.ndarray:
    """Sample a random t2 amplitudes tensor.

    Args:
        norb: The number of orbitals.
        nocc: The number of orbitals that are occupied by an electron.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.
        dtype: The data type to use for the result.

    Returns:
        The sampled t2 amplitudes tensor.
    """
    rng = np.random.default_rng(seed)
    nvrt = norb - nocc
    t2 = np.zeros((nocc, nocc, nvrt, nvrt), dtype=dtype)
    pairs = itertools.product(range(nocc), range(nocc, norb))
    for (i, a), (j, b) in itertools.combinations_with_replacement(pairs, 2):
        val = rng.standard_normal()
        t2[i, j, a - nocc, b - nocc] = val
        t2[j, i, b - nocc, a - nocc] = val
    if np.issubdtype(dtype, np.complexfloating):
        t2_large = np.zeros((norb, norb, norb, norb), dtype=dtype)
        t2_large[:nocc, :nocc, nocc:, nocc:] = t2
        orbital_rotation = random_unitary(norb, seed=rng)
        t2_large = np.einsum(
            "ijab,iI,jJ,aA,bB->IJAB",
            t2_large,
            orbital_rotation.conj(),
            orbital_rotation.conj(),
            orbital_rotation,
            orbital_rotation,
        )
        t2 = t2_large[:nocc, :nocc, nocc:, nocc:]
    return t2


def random_molecular_hamiltonian(
    norb: int, seed=None, dtype=complex
) -> hamiltonians.MolecularHamiltonian:
    """Sample a random molecular Hamiltonian.

    Args:
        norb: The number of spatial orbitals.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.
        dtype: The data type to use for the one- and two-body tensors. The constant
            term will always be of type ``float``.

    Returns:
        The sampled molecular Hamiltonian.
    """
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.complexfloating):
        one_body_tensor = random_hermitian(norb, seed=rng, dtype=dtype)
    else:
        one_body_tensor = random_real_symmetric_matrix(norb, seed=rng, dtype=dtype)
    two_body_tensor = random_two_body_tensor(norb, seed=rng, dtype=dtype)
    constant = rng.standard_normal()
    return hamiltonians.MolecularHamiltonian(
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor,
        constant=constant,
    )


def random_uccsd_restricted(
    norb: int,
    nocc: int,
    *,
    with_final_orbital_rotation: bool = False,
    real: bool = False,
    seed=None,
) -> variational.UCCSDOpRestrictedReal:
    """Sample a random UCCSD operator.

    Args:
        norb: The number of spatial orbitals.
        nocc: The number of spatial orbitals that are occupied by electrons.
        with_final_orbital_rotation: Whether to include a final orbital rotation
            in the operator.
        real: Whether to sample a real-valued object rather than a complex-valued one.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled UCCSD operator.
    """
    rng = np.random.default_rng(seed)
    dtype = float if real else complex
    nvrt = norb - nocc
    t1: np.ndarray = rng.standard_normal((nocc, nvrt)).astype(dtype, copy=False)
    if not real:
        t1 += 1j * rng.standard_normal((nocc, nvrt))
    t2 = random_t2_amplitudes(norb, nocc, seed=rng, dtype=dtype)
    final_orbital_rotation = None
    if with_final_orbital_rotation:
        unitary_func = random_orthogonal if real else random_unitary
        final_orbital_rotation = unitary_func(norb, seed=rng)
    return variational.UCCSDOpRestrictedReal(
        t1=t1, t2=t2, final_orbital_rotation=final_orbital_rotation
    )


def random_ucj_op_spin_balanced(
    norb: int,
    *,
    n_reps: int = 1,
    with_final_orbital_rotation: bool = False,
    interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]
    | None = None,
    seed=None,
) -> variational.UCJOpSpinBalanced:
    r"""Sample a random spin-balanced unitary cluster Jastrow (UCJ) operator.

    Args:
        norb: The number of spatial orbitals.
        n_reps: The number of ansatz repetitions.
        with_final_orbital_rotation: Whether to include a final orbital rotation
            in the operator.
        interaction_pairs: Optional restrictions on allowed orbital interactions
            for the diagonal Coulomb operators.
            If specified, `interaction_pairs` should be a pair of lists,
            for alpha-alpha and alpha-beta interactions, in that order.
            Either list can be substituted with ``None`` to indicate no restrictions
            on interactions.
            Each list should contain pairs of integers representing the orbitals
            that are allowed to interact. These pairs can also be interpreted as
            indices of diagonal Coulomb matrix entries that are allowed to be
            nonzero.
            Each integer pair must be upper triangular, that is, of the form
            :math:`(i, j)` where :math:`i \leq j`.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled UCJ operator.
    """
    if interaction_pairs is None:
        interaction_pairs = (None, None)
    pairs_aa, pairs_ab = interaction_pairs
    validate_interaction_pairs(pairs_aa, ordered=False)
    validate_interaction_pairs(pairs_ab, ordered=False)

    rng = np.random.default_rng(seed)
    diag_coulomb_mats = np.stack(
        [
            np.stack(
                [
                    random_real_symmetric_matrix(norb, seed=rng),
                    random_real_symmetric_matrix(norb, seed=rng),
                ]
            )
            for _ in range(n_reps)
        ]
    )
    orbital_rotations = np.stack(
        [random_unitary(norb, seed=rng) for _ in range(n_reps)]
    )
    final_orbital_rotation = None
    if with_final_orbital_rotation:
        final_orbital_rotation = random_unitary(norb, seed=rng)

    # Zero out diagonal coulomb matrix entries if requested
    if pairs_aa is not None:
        mask = np.zeros((norb, norb), dtype=bool)
        rows, cols = zip(*pairs_aa)
        mask[rows, cols] = True
        mask[cols, rows] = True
        diag_coulomb_mats[:, 0] *= mask
    if pairs_ab is not None:
        mask = np.zeros((norb, norb), dtype=bool)
        rows, cols = zip(*pairs_ab)
        mask[rows, cols] = True
        mask[cols, rows] = True
        diag_coulomb_mats[:, 1] *= mask

    return variational.UCJOpSpinBalanced(
        diag_coulomb_mats=diag_coulomb_mats,
        orbital_rotations=orbital_rotations,
        final_orbital_rotation=final_orbital_rotation,
    )


def random_ucj_op_spin_unbalanced(
    norb: int,
    *,
    n_reps: int = 1,
    interaction_pairs: tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ]
    | None = None,
    with_final_orbital_rotation: bool = False,
    seed=None,
) -> variational.UCJOpSpinUnbalanced:
    r"""Sample a random spin-unbalanced unitary cluster Jastrow (UCJ) operator.

    Args:
        norb: The number of orbitals.
        n_reps: The number of ansatz repetitions.
        interaction_pairs: Optional restrictions on allowed orbital interactions
            for the diagonal Coulomb operators.
            If specified, `interaction_pairs` should be a tuple of 3 lists,
            for alpha-alpha, alpha-beta, and beta-beta interactions, in that order.
            Any list can be substituted with ``None`` to indicate no restrictions
            on interactions.
            Each list should contain pairs of integers representing the orbitals
            that are allowed to interact. These pairs can also be interpreted as
            indices of diagonal Coulomb matrix entries that are allowed to be
            nonzero.
            For the alpha-alpha and beta-beta interactions, each integer
            pair must be upper triangular, that is, of the form :math:`(i, j)` where
            :math:`i \leq j`.
        with_final_orbital_rotation: Whether to include a final orbital rotation
            in the operator.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled UCJ operator.
    """
    if interaction_pairs is None:
        interaction_pairs = (None, None, None)
    pairs_aa, pairs_ab, pairs_bb = interaction_pairs
    validate_interaction_pairs(pairs_aa, ordered=False)
    validate_interaction_pairs(pairs_bb, ordered=True)
    validate_interaction_pairs(pairs_bb, ordered=False)

    rng = np.random.default_rng(seed)
    diag_coulomb_mats = np.stack(
        [
            np.stack(
                [
                    random_real_symmetric_matrix(norb, seed=rng),
                    rng.standard_normal((norb, norb)),
                    random_real_symmetric_matrix(norb, seed=rng),
                ]
            )
            for _ in range(n_reps)
        ]
    )
    orbital_rotations = np.stack(
        [
            np.stack([random_unitary(norb, seed=rng), random_unitary(norb, seed=rng)])
            for _ in range(n_reps)
        ]
    )
    final_orbital_rotation = None
    if with_final_orbital_rotation:
        final_orbital_rotation = np.stack(
            [random_unitary(norb, seed=rng), random_unitary(norb, seed=rng)]
        )

    # Zero out diagonal coulomb matrix entries if requested
    if pairs_aa is not None:
        mask = np.zeros((norb, norb), dtype=bool)
        rows, cols = zip(*pairs_aa)
        mask[rows, cols] = True
        mask[cols, rows] = True
        diag_coulomb_mats[:, 0] *= mask
    if pairs_ab is not None:
        mask = np.zeros((norb, norb), dtype=bool)
        rows, cols = zip(*pairs_ab)
        mask[rows, cols] = True
        diag_coulomb_mats[:, 1] *= mask
    if pairs_bb is not None:
        mask = np.zeros((norb, norb), dtype=bool)
        rows, cols = zip(*pairs_bb)
        mask[rows, cols] = True
        mask[cols, rows] = True
        diag_coulomb_mats[:, 2] *= mask

    return variational.UCJOpSpinUnbalanced(
        diag_coulomb_mats=diag_coulomb_mats,
        orbital_rotations=orbital_rotations,
        final_orbital_rotation=final_orbital_rotation,
    )


def random_ucj_op_spinless(
    norb: int,
    *,
    n_reps: int = 1,
    interaction_pairs: list[tuple[int, int]] | None = None,
    with_final_orbital_rotation: bool = False,
    seed=None,
) -> variational.UCJOpSpinless:
    r"""Sample a random spinless unitary cluster Jastrow (UCJ) operator.

    Args:
        norb: The number of orbitals.
        n_reps: The number of ansatz repetitions.
        with_final_orbital_rotation: Whether to include a final orbital rotation
            in the operator.
        interaction_pairs: Optional restrictions on allowed orbital interactions
            for the diagonal Coulomb operators.
            If specified, `interaction_pairs` should be a list of integer pairs
            representing the orbitals that are allowed to interact. These pairs
            can also be interpreted as indices of diagonal Coulomb matrix entries
            that are allowed to be nonzero.
            Each integer pair must be upper triangular, that is, of the form
            :math:`(i, j)` where :math:`i \leq j`.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled UCJ operator.
    """
    validate_interaction_pairs(interaction_pairs, ordered=False)

    rng = np.random.default_rng(seed)
    diag_coulomb_mats = np.stack(
        [random_real_symmetric_matrix(norb, seed=rng) for _ in range(n_reps)]
    )
    orbital_rotations = np.stack(
        [random_unitary(norb, seed=rng) for _ in range(n_reps)]
    )
    final_orbital_rotation = None
    if with_final_orbital_rotation:
        final_orbital_rotation = random_unitary(norb, seed=rng)

    # Zero out diagonal coulomb matrix entries if requested
    if interaction_pairs is not None:
        mask = np.zeros((norb, norb), dtype=bool)
        rows, cols = zip(*interaction_pairs)
        mask[rows, cols] = True
        mask[cols, rows] = True
        diag_coulomb_mats *= mask

    return variational.UCJOpSpinless(
        diag_coulomb_mats=diag_coulomb_mats,
        orbital_rotations=orbital_rotations,
        final_orbital_rotation=final_orbital_rotation,
    )


def random_diagonal_coulomb_hamiltonian(
    norb: int, *, real: bool = False, seed=None
) -> hamiltonians.DiagonalCoulombHamiltonian:
    """Sample a random diagonal Coulomb Hamiltonian.

    Args:
        norb: The number of spatial orbitals.
        real: Whether to sample a real-valued object rather than a complex-valued one.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled diagonal Coulomb Hamiltonian.
    """
    rng = np.random.default_rng(seed)
    if real:
        one_body_tensor = random_real_symmetric_matrix(norb, seed=rng)
    else:
        one_body_tensor = random_hermitian(norb, seed=rng)
    diag_coulomb_mat_a = random_real_symmetric_matrix(norb, seed=rng)
    diag_coulomb_mat_b = random_real_symmetric_matrix(norb, seed=rng)
    diag_coulomb_mats = np.stack([diag_coulomb_mat_a, diag_coulomb_mat_b])
    constant = rng.standard_normal()
    return hamiltonians.DiagonalCoulombHamiltonian(
        one_body_tensor, diag_coulomb_mats, constant=constant
    )


def random_double_factorized_hamiltonian(
    norb: int,
    *,
    rank: int | None = None,
    z_representation: bool = False,
    real: bool = False,
    seed=None,
) -> hamiltonians.DoubleFactorizedHamiltonian:
    """Sample a random double-factorized Hamiltonian.

    Args:
        norb: The number of spatial orbitals.
        rank: The desired number of terms in the two-body part of the Hamiltonian.
            If not specified, it will be set to ``norb * (norb + 1) // 2``.
        z_representation: Whether to return a Hamiltonian in the "Z" representation.
        real: Whether to sample a real-valued object rather than a complex-valued one.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled double-factorized Hamiltonian.
    """
    if rank is None:
        rank = norb * (norb + 1) // 2
    rng = np.random.default_rng(seed)
    if real:
        one_body_tensor = random_real_symmetric_matrix(norb, seed=rng)
        orbital_rotations = np.stack(
            [random_orthogonal(norb, seed=rng) for _ in range(rank)]
        )
    else:
        one_body_tensor = random_hermitian(norb, seed=rng)
        orbital_rotations = np.stack(
            [random_unitary(norb, seed=rng) for _ in range(rank)]
        )
    diag_coulomb_mats = np.stack(
        [random_real_symmetric_matrix(norb, seed=rng) for _ in range(rank)]
    )
    constant = rng.standard_normal()
    return hamiltonians.DoubleFactorizedHamiltonian(
        one_body_tensor=one_body_tensor,
        diag_coulomb_mats=diag_coulomb_mats,
        orbital_rotations=orbital_rotations,
        constant=constant,
        z_representation=z_representation,
    )


def random_fermion_operator(
    norb: int, n_terms: int | None = None, max_term_length: int | None = None, seed=None
) -> operators.FermionOperator:
    """Sample a random fermion operator.

    Args:
        norb: The number of spatial orbitals.
        n_terms: The number of terms to include in the operator. If not specified,
            `norb` is used.
        max_term_length: The maximum length of a term. If not specified, `norb` is used.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled fermion operator.
    """
    rng = np.random.default_rng(seed)
    if n_terms is None:
        n_terms = norb
    if max_term_length is None:
        max_term_length = norb
    coeffs: defaultdict[tuple[tuple[bool, bool, int], ...], complex] = defaultdict(
        complex
    )
    for _ in range(n_terms):
        term_length = int(rng.integers(1, max_term_length + 1))
        actions = [bool(i) for i in rng.integers(2, size=term_length)]
        spins = [bool(i) for i in rng.integers(2, size=term_length)]
        indices = [int(i) for i in rng.integers(norb, size=term_length)]
        coeff = rng.standard_normal() + 1j * rng.standard_normal()
        term = tuple(zip(actions, spins, indices))
        coeffs[term] += coeff
    return operators.FermionOperator(coeffs)


def random_fermion_hamiltonian(
    norb: int, n_terms: int | None = None, seed=None
) -> operators.FermionOperator:
    """Sample a random fermion Hamiltonian.

    A fermion Hamiltonian is hermitian and conserves particle number and spin Z.

    Args:
        norb: The number of spatial orbitals.
        n_terms: The number of terms to include in the operator. If not specified,
            `norb` is used.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled fermion Hamiltonian.
    """
    rng = np.random.default_rng(seed)
    if n_terms is None:
        n_terms = norb
    coeffs: defaultdict[tuple[tuple[bool, bool, int], ...], complex] = defaultdict(
        complex
    )
    for _ in range(n_terms):
        n_excitations = int(rng.integers(1, norb + 1))
        term = _random_num_and_spin_z_conserving_term(norb, n_excitations, seed=rng)
        term_adjoint = _adjoint_term(term)
        coeff = rng.standard_normal() + 1j * rng.standard_normal()
        coeffs[term] += coeff
        coeffs[term_adjoint] += coeff.conjugate()
    return operators.FermionOperator(coeffs)


def _random_num_and_spin_z_conserving_term(
    norb: int, n_excitations: int, seed=None
) -> tuple[tuple[bool, bool, int], ...]:
    rng = np.random.default_rng(seed)
    term = []
    for _ in range(n_excitations):
        spin = bool(rng.integers(2))
        orb_1, orb_2 = [int(x) for x in rng.integers(norb, size=2)]
        action_1, action_2 = [
            bool(x) for x in rng.choice([True, False], size=2, replace=False)
        ]
        term.append(operators.FermionAction(action_1, spin, orb_1))
        term.append(operators.FermionAction(action_2, spin, orb_2))
    return tuple(term)


def _adjoint_term(
    term: tuple[tuple[bool, bool, int], ...],
) -> tuple[tuple[bool, bool, int], ...]:
    return tuple(
        operators.FermionAction(bool(1 - action), spin, orb)
        for action, spin, orb in reversed(term)
    )
