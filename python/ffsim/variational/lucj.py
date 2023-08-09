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

import itertools
from dataclasses import dataclass

import numpy as np
import scipy.linalg

from ffsim.gates import apply_diag_coulomb_evolution, apply_orbital_rotation


def _to_mat(U, o_pairs, nb):
    U_mat = np.zeros((nb, nb))
    for m, (p, r) in enumerate(o_pairs):
        U_mat[p, r] = U[m]
    return U_mat


def _decompose(t2, no, nv, nb, o_pairs, verbose=False):
    t2_mat = np.zeros((nb**2, nb**2))
    for m, (a, i) in enumerate(o_pairs):
        for n, (b, j) in enumerate(o_pairs):
            if i < no <= a and j < no <= b:
                t2_mat[m, n] = t2[i, j, a - no, b - no]
    s, U = scipy.linalg.eigh(t2_mat)
    dec = [
        (smu, _to_mat(U[:, mu], o_pairs, nb))
        for mu, smu in enumerate(s)
        if np.abs(smu) > 1e-6
    ]
    dec = sorted(dec, key=lambda x: np.abs(x[0]))[::-1]
    if verbose:
        print("SVD of t2, singular values ", [x[0] for x in dec])
    return dec


@dataclass
class LUCJOperator:
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
    ) -> "LUCJOperator":
        triu_indices = list(itertools.combinations_with_replacement(range(norb), 2))
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
        return LUCJOperator(
            diag_coulomb_mats_alpha_alpha=diag_coulomb_mats_alpha_alpha,
            diag_coulomb_mats_alpha_beta=diag_coulomb_mats_alpha_beta,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=final_orbital_rotation,
        )

    def to_parameters(
        self,
        *,
        alpha_alpha_indices: list[int] | None = None,
        alpha_beta_indices: list[int] | None = None,
    ) -> np.ndarray:
        triu_indices = list(
            itertools.combinations_with_replacement(range(self.norb), 2)
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
        t1: np.ndarray | None = None,
        n_reps: int | None = None,
    ) -> "LUCJOperator":
        # TODO allow specifying alpha-alpha and alpha-beta indices
        nocc, _, nvrt, _ = t2.shape
        norb = nocc + nvrt
        o_pairs = list(itertools.product(range(nocc, norb), range(nocc)))
        # TODO use ffsim.linalg.double_factorized
        low_rank = _decompose(t2, nocc, nvrt, norb, o_pairs)
        diag_coulomb_mats_alpha_alpha = []
        diag_coulomb_mats_alpha_beta = []
        orbital_rotations = []
        for smu, Umu in low_rank:
            Xmu_p = (1 - 1j) / 2.0 * (Umu + 1j * Umu.T)
            Xmu_m = (1 + 1j) / 2.0 * (Umu - 1j * Umu.T)
            gmu_p, Vmu_p = scipy.linalg.eigh(Xmu_p)
            gmu_m, Vmu_m = scipy.linalg.eigh(Xmu_m)
            Jmu_p = smu * np.einsum("p,q->pq", gmu_p, gmu_p)
            Jmu_m = -smu * np.einsum("p,q->pq", gmu_m, gmu_m)
            diag_coulomb_mats_alpha_alpha.append(Jmu_p)
            diag_coulomb_mats_alpha_alpha.append(Jmu_m)
            diag_coulomb_mats_alpha_beta.append(Jmu_p)
            diag_coulomb_mats_alpha_beta.append(Jmu_m)
            orbital_rotations.append(Vmu_p)
            orbital_rotations.append(Vmu_m)
        final_orbital_rotation = None
        if t1 is not None:
            final_orbital_rotation_generator = np.zeros((norb, norb), dtype=complex)
            final_orbital_rotation_generator[:nocc, nocc:] = t1
            final_orbital_rotation_generator[nocc:, :nocc] = -t1.T
            final_orbital_rotation = scipy.linalg.expm(final_orbital_rotation_generator)
        return LUCJOperator(
            diag_coulomb_mats_alpha_alpha=np.stack(
                diag_coulomb_mats_alpha_alpha[:n_reps]
            ),
            diag_coulomb_mats_alpha_beta=np.stack(
                diag_coulomb_mats_alpha_beta[:n_reps]
            ),
            orbital_rotations=np.stack(orbital_rotations[:n_reps]),
            final_orbital_rotation=final_orbital_rotation,
        )

    def to_t_amplitudes(self, nocc: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        t2 = np.zeros((self.norb, self.norb, self.norb, self.norb), dtype=complex)
        for mu in range(self.n_reps):
            Jmu = self.diag_coulomb_mats_alpha_beta[mu]
            Vmu = self.orbital_rotations[mu]
            t2 += 1j * np.einsum(
                "pq,ap,ip,bq,jq->aibj", Jmu, Vmu, np.conj(Vmu), Vmu, np.conj(Vmu)
            )
        t1 = np.zeros((self.norb, self.norb))[:nocc, nocc:]
        if self.final_orbital_rotation is not None:
            t1 = scipy.linalg.logm(self.final_orbital_rotation)[:nocc, nocc:]
        return np.einsum("aibj->ijab", t2)[:nocc, :nocc, nocc:, nocc:], t1


def apply_lucj_operator(
    vec: np.ndarray,
    operator: LUCJOperator,
    *,
    norb: int,
    nelec: tuple[int, int],
    copy: bool = True,
):
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
            # TODO use positive time convention for consistency
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
