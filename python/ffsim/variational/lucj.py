# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""(Local) unitary cluster Jastrow ansatz."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Tuple, cast

import numpy as np
import scipy.linalg

from ffsim.gates import apply_diag_coulomb_evolution, apply_orbital_rotation


def _to_mat(vec: np.ndarray, o_pairs: list[tuple[int, int]], norb: int):
    mat = np.zeros((norb, norb))
    for val, (p, r) in zip(vec, o_pairs):
        mat[p, r] = val
    return mat


def _decompose(t2: np.ndarray, o_pairs: list[tuple[int, int]], tol: float = 1e-8):
    nocc, _, nvrt, _ = t2.shape
    norb = nocc + nvrt
    t2_mat = np.zeros((norb**2, norb**2))
    for m, (a, i) in enumerate(o_pairs):
        for n, (b, j) in enumerate(o_pairs):
            if i < nocc <= a and j < nocc <= b:
                t2_mat[m, n] = t2[i, j, a - nocc, b - nocc]
    eigs, vecs = scipy.linalg.eigh(t2_mat)
    dec = [
        (eig, _to_mat(vec, o_pairs, norb))
        for eig, vec in zip(eigs, vecs.T)
        if not np.isclose(eig, 0, atol=tol)
    ]
    return sorted(dec, key=lambda x: np.abs(x[0]))[::-1]


@dataclass
class UnitaryClusterJastrowOp:
    r"""A unitary cluster Jastrow operator.

    A unitary cluster Jastrow (UCJ) operator has the form

    .. math::

        \prod_{k = 1}^L \mathcal{W_k} e^{i \mathcal{J}_k} \mathcal{W_k^\dagger}

    where each :math:`\mathcal{W_k}` is an orbital rotation and each :math:`\mathcal{J}`
    is a diagonal Coulomb operator of the form

    .. math::

        \mathcal{J} = \frac12\sum_{ij,\sigma \tau}
        \mathbf{J}^{\sigma \tau}_{ij} n_{i,\sigma} n_{j,\tau}.

    In order that the operator commutes with the total spin operator, we enforce that
    :math:`\mathbf{J}^{\alpha\alpha} = \mathbf{J}^{\beta\beta}` and
    :math:`\mathbf{J}^{\alpha\beta} = \mathbf{J}^{\beta\alpha}`. As a result, we have
    two sets of matrices for describing the diagonal Coulomb operators:
    "alpha-alpha" matrices containing coefficients for terms involving the same spin,
    and "alpha-beta" matrices containing coefficients for terms involving different
    spins.

    To support variational optimization of the orbital basis, an optional final
    orbital rotation can be included in the operator, to be performed at the end.

    Attributes:
        diag_coulomb_mats_alpha_alpha: The "alpha-alpha" diagonal Coulomb matrices.
        diag_coulomb_mats_alpha_beta: The "alpha-beta" diagonal Coulomb matrices.
        orbital_rotations: The orbital rotations.
        final_orbital_rotation: The optional final orbital rotation.
    """

    diag_coulomb_mats_alpha_alpha: np.ndarray
    diag_coulomb_mats_alpha_beta: np.ndarray
    orbital_rotations: np.ndarray
    final_orbital_rotation: np.ndarray | None = None

    @property
    def norb(self):
        """The number of spatial orbitals."""
        return self.diag_coulomb_mats_alpha_alpha.shape[1]

    @property
    def n_reps(self):
        """The number of ansatz repetitions."""
        return self.diag_coulomb_mats_alpha_alpha.shape[0]

    @staticmethod
    def from_parameters(
        params: np.ndarray,
        *,
        norb: int,
        n_reps: int,
        alpha_alpha_indices: list[tuple[int, int]] | None = None,
        alpha_beta_indices: list[tuple[int, int]] | None = None,
        with_final_orbital_rotation: bool = False,
    ) -> "UnitaryClusterJastrowOp":
        """Initialize the UCJ operator from a real-valued parameter vector."""
        triu_indices = cast(
            List[Tuple[int, int]],
            list(itertools.combinations_with_replacement(range(norb), 2)),
        )
        triu_indices_no_diag = list(itertools.combinations(range(norb), 2))
        if alpha_alpha_indices is None:
            alpha_alpha_indices = triu_indices
        if alpha_beta_indices is None:
            alpha_beta_indices = triu_indices
        diag_coulomb_mats_alpha_alpha = np.zeros((n_reps, norb, norb))
        diag_coulomb_mats_alpha_beta = np.zeros((n_reps, norb, norb))
        orbital_rotation_generators = np.zeros((n_reps, norb, norb), dtype=complex)
        index = 0
        # diag coulomb matrices, alpha-alpha
        indices = alpha_alpha_indices
        if indices:
            n_params = len(indices)
            rows, cols = zip(*indices)
            for mat in diag_coulomb_mats_alpha_alpha:
                vals = params[index : index + n_params]
                mat[rows, cols] = vals
                mat[cols, rows] = vals
                index += n_params
        # diag coulomb matrices, alpha-beta
        indices = alpha_beta_indices
        if indices:
            n_params = len(indices)
            rows, cols = zip(*indices)
            for mat in diag_coulomb_mats_alpha_beta:
                vals = params[index : index + n_params]
                mat[rows, cols] = vals
                mat[cols, rows] = vals
                index += n_params
        # orbital rotations, imaginary part
        indices = triu_indices
        n_params = len(indices)
        rows, cols = zip(*indices)
        for mat in orbital_rotation_generators:
            vals = 1j * params[index : index + n_params]
            mat[rows, cols] = vals
            mat[cols, rows] = vals
            index += n_params
        # orbital rotations, real part
        indices = triu_indices_no_diag
        n_params = len(indices)
        rows, cols = zip(*indices)
        for mat in orbital_rotation_generators:
            vals = params[index : index + n_params]
            mat[rows, cols] += vals
            mat[cols, rows] -= vals
            index += n_params
        # exponentiate orbital rotation generators
        orbital_rotations = np.stack(
            [scipy.linalg.expm(mat) for mat in orbital_rotation_generators]
        )
        # final orbital rotation
        final_orbital_rotation = None
        if with_final_orbital_rotation:
            final_orbital_rotation_generator = np.zeros((norb, norb), dtype=complex)
            # final orbital rotation, imaginary part
            indices = triu_indices
            n_params = len(indices)
            rows, cols = zip(*indices)
            vals = 1j * params[index : index + n_params]
            final_orbital_rotation_generator[rows, cols] = vals
            final_orbital_rotation_generator[cols, rows] = vals
            index += n_params
            # final orbital rotation, real part
            indices = triu_indices_no_diag
            n_params = len(indices)
            rows, cols = zip(*indices)
            vals = params[index : index + n_params]
            final_orbital_rotation_generator[rows, cols] += vals
            final_orbital_rotation_generator[cols, rows] -= vals
            # exponentiate final orbital rotation generator
            final_orbital_rotation = scipy.linalg.expm(final_orbital_rotation_generator)
        return UnitaryClusterJastrowOp(
            diag_coulomb_mats_alpha_alpha=diag_coulomb_mats_alpha_alpha,
            diag_coulomb_mats_alpha_beta=diag_coulomb_mats_alpha_beta,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=final_orbital_rotation,
        )

    def to_parameters(
        self,
        *,
        alpha_alpha_indices: list[tuple[int, int]] | None = None,
        alpha_beta_indices: list[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        """Convert the UCJ operator to a real-valued parameter vector."""
        triu_indices = cast(
            List[Tuple[int, int]],
            list(itertools.combinations_with_replacement(range(self.norb), 2)),
        )
        triu_indices_no_diag = list(itertools.combinations(range(self.norb), 2))
        if alpha_alpha_indices is None:
            alpha_alpha_indices = triu_indices
        if alpha_beta_indices is None:
            alpha_beta_indices = triu_indices
        ntheta = self.n_reps * (
            len(alpha_alpha_indices) + len(alpha_beta_indices) + self.norb**2
        )
        if self.final_orbital_rotation is not None:
            ntheta += self.norb**2
        orbital_rotation_generators = [
            scipy.linalg.logm(mat) for mat in self.orbital_rotations
        ]
        theta = np.zeros(ntheta)
        index = 0
        # diag coulomb matrices, alpha-alpha
        indices = alpha_alpha_indices
        if indices:
            n_params = len(indices)
            for mat in self.diag_coulomb_mats_alpha_alpha:
                theta[index : index + n_params] = mat[tuple(zip(*indices))]
                index += n_params
        # diag coulomb matrices, alpha-beta
        indices = alpha_beta_indices
        if indices:
            n_params = len(indices)
            for mat in self.diag_coulomb_mats_alpha_beta:
                theta[index : index + n_params] = mat[tuple(zip(*indices))]
                index += n_params
        # orbital rotations, imaginary part
        indices = triu_indices
        n_params = len(indices)
        for mat in orbital_rotation_generators:
            theta[index : index + n_params] = mat[tuple(zip(*indices))].imag
            index += n_params
        # orbital rotations, real part
        indices = triu_indices_no_diag
        n_params = len(indices)
        for mat in orbital_rotation_generators:
            theta[index : index + n_params] = mat[tuple(zip(*indices))].real
            index += n_params
        # final orbital rotation
        if self.final_orbital_rotation is not None:
            mat = scipy.linalg.logm(self.final_orbital_rotation)
            # final orbital rotation, imaginary part
            indices = triu_indices
            n_params = len(indices)
            theta[index : index + n_params] = mat[tuple(zip(*indices))].imag
            index += n_params
            # final orbital rotation, real part
            indices = triu_indices_no_diag
            n_params = len(indices)
            theta[index : index + n_params] = mat[tuple(zip(*indices))].real
        return theta

    @staticmethod
    def from_t_amplitudes(
        t2: np.ndarray,
        *,
        n_reps: int | None = None,
        t1: np.ndarray | None = None,
    ) -> "UnitaryClusterJastrowOp":
        """Initialize the UCJ operator from t2 (and optionally t1) amplitudes."""
        # TODO maybe allow specifying alpha-alpha and alpha-beta indices
        nocc, _, nvrt, _ = t2.shape
        norb = nocc + nvrt
        o_pairs = list(itertools.product(range(nocc, norb), range(nocc)))
        # TODO use ffsim.linalg.double_factorized
        low_rank = _decompose(t2, o_pairs)
        diag_coulomb_mats_alpha_alpha = []
        diag_coulomb_mats_alpha_beta = []
        orbital_rotations = []
        for smu, Umu in low_rank:
            Xmu = 0.5 * (1 - 1j) * (Umu + 1j * Umu.T)
            gmu, Vmu = scipy.linalg.eigh(Xmu)
            Jmu = smu * np.outer(gmu, gmu)
            diag_coulomb_mats_alpha_alpha.append(Jmu)
            diag_coulomb_mats_alpha_alpha.append(-Jmu)
            diag_coulomb_mats_alpha_beta.append(Jmu)
            diag_coulomb_mats_alpha_beta.append(-Jmu)
            orbital_rotations.append(Vmu)
            orbital_rotations.append(Vmu.conj())
        final_orbital_rotation = None
        if t1 is not None:
            final_orbital_rotation_generator = np.zeros((norb, norb), dtype=complex)
            final_orbital_rotation_generator[:nocc, nocc:] = t1
            final_orbital_rotation_generator[nocc:, :nocc] = -t1.T
            final_orbital_rotation = scipy.linalg.expm(final_orbital_rotation_generator)
        return UnitaryClusterJastrowOp(
            diag_coulomb_mats_alpha_alpha=np.stack(
                diag_coulomb_mats_alpha_alpha[:n_reps]
            ),
            diag_coulomb_mats_alpha_beta=np.stack(
                diag_coulomb_mats_alpha_beta[:n_reps]
            ),
            orbital_rotations=np.stack(orbital_rotations[:n_reps]),
            final_orbital_rotation=final_orbital_rotation,
        )

    def to_t_amplitudes(
        self, nocc: int | None = None
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Convert the UCJ operator to t2 (and possibly t1) amplitudes."""
        t2 = np.zeros((self.norb, self.norb, self.norb, self.norb), dtype=complex)
        for mu in range(self.n_reps):
            Jmu = self.diag_coulomb_mats_alpha_beta[mu]
            Vmu = self.orbital_rotations[mu]
            t2 += 1j * np.einsum(
                "pq,ap,ip,bq,jq->aibj", Jmu, Vmu, np.conj(Vmu), Vmu, np.conj(Vmu)
            )
        t2 = np.einsum("aibj->ijab", t2)[:nocc, :nocc, nocc:, nocc:]

        if self.final_orbital_rotation is None:
            return t2

        t1 = scipy.linalg.logm(self.final_orbital_rotation)[:nocc, nocc:]
        return t2, t1


def apply_unitary_cluster_jastrow_op(
    vec: np.ndarray,
    operator: UnitaryClusterJastrowOp,
    *,
    norb: int,
    nelec: tuple[int, int],
    copy: bool = True,
) -> np.ndarray:
    """Apply a UCJ operator to a vector.

    Args:
        vec: The state vector to be transformed.
        operator: The UCJ operator.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
            vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
            vector, but the original vector may have its data overwritten.
            It is also possible that the original vector is returned,
            modified in-place.
    """
    if copy:
        vec = vec.copy()
    for mat, mat_alpha_beta, orbital_rotation in zip(
        operator.diag_coulomb_mats_alpha_alpha,
        operator.diag_coulomb_mats_alpha_beta,
        operator.orbital_rotations,
    ):
        vec = apply_diag_coulomb_evolution(
            vec,
            mat=mat,
            time=-1.0,
            norb=norb,
            nelec=nelec,
            mat_alpha_beta=mat_alpha_beta,
            orbital_rotation=orbital_rotation,
            copy=False,
        )
    if operator.final_orbital_rotation is not None:
        vec = apply_orbital_rotation(
            vec,
            mat=operator.final_orbital_rotation,
            norb=norb,
            nelec=nelec,
            copy=False,
        )
    return vec
