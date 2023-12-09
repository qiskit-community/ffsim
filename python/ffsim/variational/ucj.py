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
from opt_einsum import contract

from ffsim.gates import apply_diag_coulomb_evolution, apply_orbital_rotation
from ffsim.linalg import double_factorized_t2
from ffsim.variational.util import (
    orbital_rotation_from_parameters,
    orbital_rotation_to_parameters,
)


@dataclass(frozen=True)
class UCJOperator:
    r"""A unitary cluster Jastrow operator.

    A unitary cluster Jastrow (UCJ) operator has the form

    .. math::

        \prod_{k = 1}^L \mathcal{W_k} e^{i \mathcal{J}_k} \mathcal{W_k^\dagger}

    where each :math:`\mathcal{W_k}` is an orbital rotation and each :math:`\mathcal{J}`
    is a diagonal Coulomb operator of the form

    .. math::

        \mathcal{J} = \frac12\sum_{\sigma \tau, ij}
        \mathbf{J}^{\sigma \tau}_{ij} n_{\sigma, i} n_{\tau, j}.

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
    ) -> UCJOperator:
        """Initialize the UCJ operator from a real-valued parameter vector."""
        return UCJOperator(
            *_ucj_from_parameters(
                params,
                norb=norb,
                n_reps=n_reps,
                alpha_alpha_indices=alpha_alpha_indices,
                alpha_beta_indices=alpha_beta_indices,
                with_final_orbital_rotation=with_final_orbital_rotation,
            )
        )

    def to_parameters(
        self,
        *,
        alpha_alpha_indices: list[tuple[int, int]] | None = None,
        alpha_beta_indices: list[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        """Convert the UCJ operator to a real-valued parameter vector."""
        return _ucj_to_parameters(
            diag_coulomb_mats_alpha_alpha=self.diag_coulomb_mats_alpha_alpha,
            diag_coulomb_mats_alpha_beta=self.diag_coulomb_mats_alpha_beta,
            orbital_rotations=self.orbital_rotations,
            final_orbital_rotation=self.final_orbital_rotation,
            alpha_alpha_indices=alpha_alpha_indices,
            alpha_beta_indices=alpha_beta_indices,
        )

    @staticmethod
    def from_t_amplitudes(
        t2_amplitudes: np.ndarray,
        *,
        t1_amplitudes: np.ndarray | None = None,
        n_reps: int | None = None,
        tol: float = 1e-8,
    ) -> UCJOperator:
        """Initialize the UCJ operator from t2 (and optionally t1) amplitudes."""
        # TODO maybe allow specifying alpha-alpha and alpha-beta indices
        nocc, _, nvrt, _ = t2_amplitudes.shape
        norb = nocc + nvrt

        diag_coulomb_mats, orbital_rotations = double_factorized_t2(
            t2_amplitudes, tol=tol
        )
        n_vecs, norb, _ = diag_coulomb_mats.shape
        expanded_diag_coulomb_mats = np.zeros((2 * n_vecs, norb, norb))
        expanded_orbital_rotations = np.zeros((2 * n_vecs, norb, norb), dtype=complex)
        expanded_diag_coulomb_mats[::2] = diag_coulomb_mats
        expanded_diag_coulomb_mats[1::2] = -diag_coulomb_mats
        expanded_orbital_rotations[::2] = orbital_rotations
        expanded_orbital_rotations[1::2] = orbital_rotations.conj()

        final_orbital_rotation = None
        if t1_amplitudes is not None:
            final_orbital_rotation_generator = np.zeros((norb, norb), dtype=complex)
            final_orbital_rotation_generator[:nocc, nocc:] = t1_amplitudes
            final_orbital_rotation_generator[nocc:, :nocc] = -t1_amplitudes.T
            final_orbital_rotation = scipy.linalg.expm(final_orbital_rotation_generator)

        return UCJOperator(
            diag_coulomb_mats_alpha_alpha=expanded_diag_coulomb_mats[:n_reps],
            diag_coulomb_mats_alpha_beta=expanded_diag_coulomb_mats[:n_reps],
            orbital_rotations=expanded_orbital_rotations[:n_reps],
            final_orbital_rotation=final_orbital_rotation,
        )

    def to_t_amplitudes(
        self, nocc: int | None = None
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Convert the UCJ operator to t2 (and possibly t1) amplitudes."""
        # TODO this ignores diag_coulomb_mats_alpha_beta
        t2 = (
            1j
            * contract(
                "kpq,kap,kip,kbq,kjq->ijab",
                self.diag_coulomb_mats_alpha_alpha,
                self.orbital_rotations,
                self.orbital_rotations.conj(),
                self.orbital_rotations,
                self.orbital_rotations.conj(),
                optimize="greedy",
            )[:nocc, :nocc, nocc:, nocc:]
        )

        if self.final_orbital_rotation is None:
            return t2

        t1 = scipy.linalg.logm(self.final_orbital_rotation)[:nocc, nocc:]
        return t2, t1

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: tuple[int, int], copy: bool
    ) -> np.ndarray:
        """Apply the operator to a vector."""
        if copy:
            vec = vec.copy()
        for mat, mat_alpha_beta, orbital_rotation in zip(
            self.diag_coulomb_mats_alpha_alpha,
            self.diag_coulomb_mats_alpha_beta,
            self.orbital_rotations,
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
        if self.final_orbital_rotation is not None:
            vec = apply_orbital_rotation(
                vec,
                mat=self.final_orbital_rotation,
                norb=norb,
                nelec=nelec,
                copy=False,
            )
        return vec


@dataclass
class RealUCJOperator:
    r"""Real-valued unitary cluster Jastrow operator.

    A real-valued unitary cluster Jastrow (UCJ) operator has the form

    .. math::

        \prod_{k = 1}^L
        \mathcal{W_k^*} e^{i \mathcal{-J}_k} \mathcal{W_k}^T
        \mathcal{W_k} e^{i \mathcal{J}_k} \mathcal{W_k^\dagger}

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
    ) -> "RealUCJOperator":
        """Initialize the real UCJ operator from a real-valued parameter vector."""
        return RealUCJOperator(
            *_ucj_from_parameters(
                params,
                norb=norb,
                n_reps=n_reps,
                alpha_alpha_indices=alpha_alpha_indices,
                alpha_beta_indices=alpha_beta_indices,
                with_final_orbital_rotation=with_final_orbital_rotation,
            )
        )

    def to_parameters(
        self,
        *,
        alpha_alpha_indices: list[tuple[int, int]] | None = None,
        alpha_beta_indices: list[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        """Convert the real UCJ operator to a real-valued parameter vector."""
        return _ucj_to_parameters(
            diag_coulomb_mats_alpha_alpha=self.diag_coulomb_mats_alpha_alpha,
            diag_coulomb_mats_alpha_beta=self.diag_coulomb_mats_alpha_beta,
            orbital_rotations=self.orbital_rotations,
            final_orbital_rotation=self.final_orbital_rotation,
            alpha_alpha_indices=alpha_alpha_indices,
            alpha_beta_indices=alpha_beta_indices,
        )

    @staticmethod
    def from_t_amplitudes(
        t2_amplitudes: np.ndarray,
        *,
        t1_amplitudes: np.ndarray | None = None,
        n_reps: int | None = None,
        tol: float = 1e-8,
    ) -> "RealUCJOperator":
        """Initialize the real UCJ operator from t2 (and optionally t1) amplitudes."""
        # TODO maybe allow specifying alpha-alpha and alpha-beta indices
        nocc, _, nvrt, _ = t2_amplitudes.shape
        norb = nocc + nvrt

        diag_coulomb_mats, orbital_rotations = double_factorized_t2(
            t2_amplitudes, tol=tol
        )
        _, norb, _ = diag_coulomb_mats.shape

        final_orbital_rotation = None
        if t1_amplitudes is not None:
            final_orbital_rotation_generator = np.zeros((norb, norb), dtype=complex)
            final_orbital_rotation_generator[:nocc, nocc:] = t1_amplitudes
            final_orbital_rotation_generator[nocc:, :nocc] = -t1_amplitudes.T
            final_orbital_rotation = scipy.linalg.expm(final_orbital_rotation_generator)

        return RealUCJOperator(
            diag_coulomb_mats_alpha_alpha=diag_coulomb_mats[:n_reps],
            diag_coulomb_mats_alpha_beta=diag_coulomb_mats[:n_reps],
            orbital_rotations=orbital_rotations[:n_reps],
            final_orbital_rotation=final_orbital_rotation,
        )

    def to_t_amplitudes(
        self, nocc: int | None = None
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Convert the UCJ operator to t2 (and possibly t1) amplitudes."""
        # TODO this ignores diag_coulomb_mats_alpha_beta
        t2 = 1j * (
            contract(
                "kpq,kap,kip,kbq,kjq->ijab",
                self.diag_coulomb_mats_alpha_alpha,
                self.orbital_rotations,
                self.orbital_rotations.conj(),
                self.orbital_rotations,
                self.orbital_rotations.conj(),
                optimize="greedy",
            )[:nocc, :nocc, nocc:, nocc:]
            + contract(
                "kpq,kap,kip,kbq,kjq->ijab",
                -self.diag_coulomb_mats_alpha_alpha,
                self.orbital_rotations.conj(),
                self.orbital_rotations,
                self.orbital_rotations.conj(),
                self.orbital_rotations,
                optimize="greedy",
            )[:nocc, :nocc, nocc:, nocc:]
        )

        if self.final_orbital_rotation is None:
            return t2

        t1 = scipy.linalg.logm(self.final_orbital_rotation)[:nocc, nocc:]
        return t2, t1

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: tuple[int, int], copy: bool
    ) -> np.ndarray:
        """Apply the operator to a vector."""
        if copy:
            vec = vec.copy()
        for mat, mat_alpha_beta, orbital_rotation in zip(
            self.diag_coulomb_mats_alpha_alpha,
            self.diag_coulomb_mats_alpha_beta,
            self.orbital_rotations,
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
            vec = apply_diag_coulomb_evolution(
                vec,
                mat=-mat,
                time=-1.0,
                norb=norb,
                nelec=nelec,
                mat_alpha_beta=-mat_alpha_beta,
                orbital_rotation=orbital_rotation.conj(),
                copy=False,
            )
        if self.final_orbital_rotation is not None:
            vec = apply_orbital_rotation(
                vec,
                mat=self.final_orbital_rotation,
                norb=norb,
                nelec=nelec,
                copy=False,
            )
        return vec


def _ucj_from_parameters(
    params: np.ndarray,
    *,
    norb: int,
    n_reps: int,
    alpha_alpha_indices: list[tuple[int, int]] | None = None,
    alpha_beta_indices: list[tuple[int, int]] | None = None,
    with_final_orbital_rotation: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    triu_indices = cast(
        List[Tuple[int, int]],
        list(itertools.combinations_with_replacement(range(norb), 2)),
    )
    if alpha_alpha_indices is None:
        alpha_alpha_indices = triu_indices
    if alpha_beta_indices is None:
        alpha_beta_indices = triu_indices
    diag_coulomb_mats_alpha_alpha = np.zeros((n_reps, norb, norb))
    diag_coulomb_mats_alpha_beta = np.zeros((n_reps, norb, norb))
    orbital_rotations = np.zeros((n_reps, norb, norb), dtype=complex)
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
    # orbital rotations
    for mat in orbital_rotations:
        mat[:] = orbital_rotation_from_parameters(
            params[index : index + norb**2], norb
        )
        index += norb**2
    # final orbital rotation
    final_orbital_rotation = None
    if with_final_orbital_rotation:
        final_orbital_rotation = orbital_rotation_from_parameters(params[index:], norb)
    return (
        diag_coulomb_mats_alpha_alpha,
        diag_coulomb_mats_alpha_beta,
        orbital_rotations,
        final_orbital_rotation,
    )


def _ucj_to_parameters(
    diag_coulomb_mats_alpha_alpha: np.ndarray,
    diag_coulomb_mats_alpha_beta: np.ndarray,
    orbital_rotations: np.ndarray,
    final_orbital_rotation: np.ndarray | None = None,
    *,
    alpha_alpha_indices: list[tuple[int, int]] | None = None,
    alpha_beta_indices: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    n_reps, norb, _ = diag_coulomb_mats_alpha_alpha.shape
    triu_indices = cast(
        List[Tuple[int, int]],
        list(itertools.combinations_with_replacement(range(norb), 2)),
    )
    if alpha_alpha_indices is None:
        alpha_alpha_indices = triu_indices
    if alpha_beta_indices is None:
        alpha_beta_indices = triu_indices
    ntheta = n_reps * (len(alpha_alpha_indices) + len(alpha_beta_indices) + norb**2)
    if final_orbital_rotation is not None:
        ntheta += norb**2
    theta = np.zeros(ntheta)
    index = 0
    # diag coulomb matrices, alpha-alpha
    indices = alpha_alpha_indices
    if indices:
        n_params = len(indices)
        for mat in diag_coulomb_mats_alpha_alpha:
            theta[index : index + n_params] = mat[tuple(zip(*indices))]
            index += n_params
    # diag coulomb matrices, alpha-beta
    indices = alpha_beta_indices
    if indices:
        n_params = len(indices)
        for mat in diag_coulomb_mats_alpha_beta:
            theta[index : index + n_params] = mat[tuple(zip(*indices))]
            index += n_params
    # orbital rotations
    for orbital_rotation in orbital_rotations:
        theta[index : index + norb**2] = orbital_rotation_to_parameters(
            orbital_rotation
        )
        index += norb**2
    # final orbital rotation
    if final_orbital_rotation is not None:
        theta[index : index + norb**2] = orbital_rotation_to_parameters(
            final_orbital_rotation
        )
    return theta
