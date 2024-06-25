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
from typing import cast

import numpy as np
import scipy.linalg
from opt_einsum import contract
from typing_extensions import deprecated

from ffsim.gates import apply_diag_coulomb_evolution, apply_orbital_rotation
from ffsim.linalg import double_factorized_t2
from ffsim.variational.util import (
    orbital_rotation_from_parameters,
    orbital_rotation_to_parameters,
)


def _validate_diag_coulomb_indices(indices: list[tuple[int, int]] | None):
    if indices is not None:
        for i, j in indices:
            if i > j:
                raise ValueError(
                    "When specifying diagonal Coulomb indices, you must give only "
                    "upper trianglular indices. "
                    f"Got {(i, j)}, which is a lower triangular index."
                )


@dataclass(frozen=True)
@deprecated("The UCJOperator class is deprecated. Use UCJOpSpinBalanced instead.")
class UCJOperator:
    r"""A unitary cluster Jastrow operator.

    .. warning::
        The UCJOperator class is deprecated. Use :class:`ffsim.UCJOpSpinBalanced`
        instead.

    A unitary cluster Jastrow (UCJ) operator has the form

    .. math::

        \prod_{k = 1}^L \mathcal{W}_k e^{i \mathcal{J}_k} \mathcal{W}_k^\dagger

    where each :math:`\mathcal{W_k}` is an orbital rotation and each :math:`\mathcal{J}`
    is a diagonal Coulomb operator of the form

    .. math::

        \mathcal{J} = \frac12\sum_{\sigma \tau, ij}
        \mathbf{J}^{\sigma \tau}_{ij} n_{\sigma, i} n_{\tau, j}.

    In order that the operator commutes with the total spin Z operator, we enforce that
    :math:`\mathbf{J}^{\alpha\alpha} = \mathbf{J}^{\beta\beta}` and
    :math:`\mathbf{J}^{\alpha\beta} = \mathbf{J}^{\beta\alpha}`. As a result, we have
    two sets of matrices for describing the diagonal Coulomb operators:
    "alpha-alpha" matrices containing coefficients for terms involving the same spin,
    and "alpha-beta" matrices containing coefficients for terms involving different
    spins.

    To support variational optimization of the orbital basis, an optional final
    orbital rotation can be included in the operator, to be performed at the end.

    Attributes:
        diag_coulomb_mats_alpha_alpha (np.ndarray): The "alpha-alpha" diagonal Coulomb
            matrices.
        diag_coulomb_mats_alpha_beta (np.ndarray): The "alpha-beta" diagonal Coulomb
            matrices.
        orbital_rotations (np.ndarray): The orbital rotations.
        final_orbital_rotation (np.ndarray): The optional final orbital rotation.
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
    def n_params(
        norb: int,
        n_reps: int,
        *,
        alpha_alpha_indices: list[tuple[int, int]] | None = None,
        alpha_beta_indices: list[tuple[int, int]] | None = None,
        with_final_orbital_rotation: bool = False,
    ) -> int:
        """Return the number of parameters of an ansatz with given settings."""
        _validate_diag_coulomb_indices(alpha_alpha_indices)
        _validate_diag_coulomb_indices(alpha_beta_indices)
        n_params_aa = (
            norb * (norb + 1) // 2
            if alpha_alpha_indices is None
            else len(alpha_alpha_indices)
        )
        n_params_ab = (
            norb * (norb + 1) // 2
            if alpha_beta_indices is None
            else len(alpha_beta_indices)
        )
        return (
            n_reps * (norb**2 + n_params_aa + n_params_ab)
            + with_final_orbital_rotation * norb**2
        )

    @staticmethod
    @deprecated("The UCJOperator class is deprecated. Use UCJOpSpinBalanced instead.")
    def from_parameters(
        params: np.ndarray,
        *,
        norb: int,
        n_reps: int,
        alpha_alpha_indices: list[tuple[int, int]] | None = None,
        alpha_beta_indices: list[tuple[int, int]] | None = None,
        with_final_orbital_rotation: bool = False,
    ) -> UCJOperator:
        r"""Initialize the UCJ operator from a real-valued parameter vector.

        Args:
            params: The real-valued parameter vector.
            norb: The number of spatial orbitals.
            n_reps: The number of ansatz repetitions (:math:`L` from the docstring of
                this class).
            alpha_alpha_indices: Allowed indices for nonzero values of the "alpha-alpha"
                diagonal Coulomb matrices (see the docstring of this class).
                If not specified, all matrix entries are allowed to be nonzero.
                This list should contain only upper trianglular indices, i.e.,
                pairs :math:`(i, j)` where :math:`i \leq j`. Passing a list with
                lower triangular indices will raise an error.
            alpha_beta_indices: Allowed indices for nonzero values of the "alpha-beta"
                diagonal Coulomb matrices (see the docstring of this class).
                If not specified, all matrix entries are allowed to be nonzero.
                This list should contain only upper trianglular indices, i.e.,
                pairs :math:`(i, j)` where :math:`i \leq j`. Passing a list with
                lower triangular indices will raise an error.
            with_final_orbital_rotation: Whether to include a final orbital rotation
                in the operator.

        Returns:
            The UCJ operator constructed from the given parameters.

        Raises:
            ValueError: alpha_alpha_indices contains lower triangular indices.
            ValueError: alpha_beta_indices contains lower triangular indices.
        """
        _validate_diag_coulomb_indices(alpha_alpha_indices)
        _validate_diag_coulomb_indices(alpha_beta_indices)
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
        r"""Convert the UCJ operator to a real-valued parameter vector.

        If `alpha_alpha_indices` or `alpha_beta_indices` is specified, the returned
        parameter vector will incorporate only the diagonal Coulomb matrix entries
        corresponding to the given indices, so the original operator will not be
        recoverable from the parameter vector.

        Args:
            alpha_alpha_indices: Allowed indices for nonzero values of the "alpha-alpha"
                diagonal Coulomb matrices (see the docstring of this class).
                If not specified, all matrix entries are allowed to be nonzero.
                This list should contain only upper trianglular indices, i.e.,
                pairs :math:`(i, j)` where :math:`i \leq j`. Passing a list with
                lower triangular indices will raise an error.
            alpha_beta_indices: Allowed indices for nonzero values of the "alpha-beta"
                diagonal Coulomb matrices (see the docstring of this class).
                If not specified, all matrix entries are allowed to be nonzero.
                This list should contain only upper trianglular indices, i.e.,
                pairs :math:`(i, j)` where :math:`i \leq j`. Passing a list with
                lower triangular indices will raise an error.

        Returns:
            The real-valued parameter vector.

        Raises:
            ValueError: alpha_alpha_indices contains lower triangular indices.
            ValueError: alpha_beta_indices contains lower triangular indices.
        """
        _validate_diag_coulomb_indices(alpha_alpha_indices)
        _validate_diag_coulomb_indices(alpha_beta_indices)
        return _ucj_to_parameters(
            diag_coulomb_mats_alpha_alpha=self.diag_coulomb_mats_alpha_alpha,
            diag_coulomb_mats_alpha_beta=self.diag_coulomb_mats_alpha_beta,
            orbital_rotations=self.orbital_rotations,
            final_orbital_rotation=self.final_orbital_rotation,
            alpha_alpha_indices=alpha_alpha_indices,
            alpha_beta_indices=alpha_beta_indices,
        )

    @staticmethod
    @deprecated("The UCJOperator class is deprecated. Use UCJOpSpinBalanced instead.")
    def from_t_amplitudes(
        t2: np.ndarray,
        *,
        t1: np.ndarray | None = None,
        n_reps: int | None = None,
        tol: float = 1e-8,
    ) -> UCJOperator:
        """Initialize the UCJ operator from t2 (and optionally t1) amplitudes."""
        nocc, _, nvrt, _ = t2.shape
        norb = nocc + nvrt

        diag_coulomb_mats, orbital_rotations = double_factorized_t2(t2, tol=tol)
        diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)[:n_reps]
        orbital_rotations = orbital_rotations.reshape(-1, norb, norb)[:n_reps]

        final_orbital_rotation = None
        if t1 is not None:
            final_orbital_rotation_generator = np.zeros((norb, norb), dtype=complex)
            final_orbital_rotation_generator[:nocc, nocc:] = t1
            final_orbital_rotation_generator[nocc:, :nocc] = -t1.T
            final_orbital_rotation = scipy.linalg.expm(final_orbital_rotation_generator)

        return UCJOperator(
            diag_coulomb_mats_alpha_alpha=diag_coulomb_mats,
            diag_coulomb_mats_alpha_beta=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=final_orbital_rotation,
        )

    def to_t_amplitudes(
        self, nocc: int | None = None
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Convert the UCJ operator to t2 (and possibly t1) amplitudes."""
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
        self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        """Apply the operator to a vector."""
        if isinstance(nelec, int):
            return NotImplemented
        if copy:
            vec = vec.copy()
        for mat, mat_alpha_beta, orbital_rotation in zip(
            self.diag_coulomb_mats_alpha_alpha,
            self.diag_coulomb_mats_alpha_beta,
            self.orbital_rotations,
        ):
            vec = apply_diag_coulomb_evolution(
                vec,
                mat=(mat, mat_alpha_beta, mat),
                time=-1.0,
                norb=norb,
                nelec=nelec,
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
@deprecated("The RealUCJOperator class is deprecated. Use UCJOpSpinBalanced instead.")
class RealUCJOperator:
    r"""Real-valued unitary cluster Jastrow operator.

    .. warning::
        The RealUCJOperator class is deprecated. Use :class:`ffsim.UCJOpSpinBalanced`
        instead.

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

    In order that the operator commutes with the total spin Z operator, we enforce that
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
    def n_params(
        norb: int,
        n_reps: int,
        *,
        alpha_alpha_indices: list[tuple[int, int]] | None = None,
        alpha_beta_indices: list[tuple[int, int]] | None = None,
        with_final_orbital_rotation: bool = False,
    ) -> int:
        """Return the number of parameters of an ansatz with given settings."""
        n_params_aa = (
            norb * (norb + 1) // 2
            if alpha_alpha_indices is None
            else len(alpha_alpha_indices)
        )
        n_params_ab = (
            norb * (norb + 1) // 2
            if alpha_beta_indices is None
            else len(alpha_beta_indices)
        )
        return (
            n_reps * (norb**2 + n_params_aa + n_params_ab)
            + with_final_orbital_rotation * norb**2
        )

    @staticmethod
    @deprecated(
        "The RealUCJOperator class is deprecated. Use UCJOpSpinBalanced instead."
    )
    def from_parameters(
        params: np.ndarray,
        *,
        norb: int,
        n_reps: int,
        alpha_alpha_indices: list[tuple[int, int]] | None = None,
        alpha_beta_indices: list[tuple[int, int]] | None = None,
        with_final_orbital_rotation: bool = False,
    ) -> "RealUCJOperator":
        r"""Initialize the real UCJ operator from a real-valued parameter vector.

        Args:
            params: The real-valued parameter vector.
            norb: The number of spatial orbitals.
            n_reps: The number of ansatz repetitions (:math:`L` from the docstring of
                this class).
            alpha_alpha_indices: Allowed indices for nonzero values of the "alpha-alpha"
                diagonal Coulomb matrices (see the docstring of this class).
                If not specified, all matrix entries are allowed to be nonzero.
                This list should contain only upper trianglular indices, i.e.,
                pairs :math:`(i, j)` where :math:`i \leq j`. Passing a list with
                lower triangular indices will raise an error.
            alpha_beta_indices: Allowed indices for nonzero values of the "alpha-beta"
                diagonal Coulomb matrices (see the docstring of this class).
                If not specified, all matrix entries are allowed to be nonzero.
                This list should contain only upper trianglular indices, i.e.,
                pairs :math:`(i, j)` where :math:`i \leq j`. Passing a list with
                lower triangular indices will raise an error.
            with_final_orbital_rotation: Whether to include a final orbital rotation
                in the operator.

        Returns:
            The real UCJ operator constructed from the given parameters.

        Raises:
            ValueError: alpha_alpha_indices contains lower triangular indices.
            ValueError: alpha_beta_indices contains lower triangular indices.
        """
        _validate_diag_coulomb_indices(alpha_alpha_indices)
        _validate_diag_coulomb_indices(alpha_beta_indices)
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
        r"""Convert the real UCJ operator to a real-valued parameter vector.

        If `alpha_alpha_indices` or `alpha_beta_indices` is specified, the returned
        parameter vector will incorporate only the diagonal Coulomb matrix entries
        corresponding to the given indices, so the original operator will not be
        recoverable from the parameter vector.

        Args:
            alpha_alpha_indices: Allowed indices for nonzero values of the "alpha-alpha"
                diagonal Coulomb matrices (see the docstring of this class).
                If not specified, all matrix entries are allowed to be nonzero.
                This list should contain only upper trianglular indices, i.e.,
                pairs :math:`(i, j)` where :math:`i \leq j`. Passing a list with
                lower triangular indices will raise an error.
            alpha_beta_indices: Allowed indices for nonzero values of the "alpha-beta"
                diagonal Coulomb matrices (see the docstring of this class).
                If not specified, all matrix entries are allowed to be nonzero.
                This list should contain only upper trianglular indices, i.e.,
                pairs :math:`(i, j)` where :math:`i \leq j`. Passing a list with
                lower triangular indices will raise an error.

        Returns:
            The real-valued parameter vector.

        Raises:
            ValueError: alpha_alpha_indices contains lower triangular indices.
            ValueError: alpha_beta_indices contains lower triangular indices.
        """
        _validate_diag_coulomb_indices(alpha_alpha_indices)
        _validate_diag_coulomb_indices(alpha_beta_indices)
        return _ucj_to_parameters(
            diag_coulomb_mats_alpha_alpha=self.diag_coulomb_mats_alpha_alpha,
            diag_coulomb_mats_alpha_beta=self.diag_coulomb_mats_alpha_beta,
            orbital_rotations=self.orbital_rotations,
            final_orbital_rotation=self.final_orbital_rotation,
            alpha_alpha_indices=alpha_alpha_indices,
            alpha_beta_indices=alpha_beta_indices,
        )

    @staticmethod
    @deprecated(
        "The RealUCJOperator class is deprecated. Use UCJOpSpinBalanced instead."
    )
    def from_t_amplitudes(
        t2: np.ndarray,
        *,
        t1: np.ndarray | None = None,
        n_reps: int | None = None,
        tol: float = 1e-8,
    ) -> "RealUCJOperator":
        """Initialize the real UCJ operator from t2 (and optionally t1) amplitudes."""
        nocc, _, nvrt, _ = t2.shape
        norb = nocc + nvrt

        diag_coulomb_mats, orbital_rotations = double_factorized_t2(t2, tol=tol)
        diag_coulomb_mats = diag_coulomb_mats[:n_reps, 0]
        orbital_rotations = orbital_rotations[:n_reps, 0]

        final_orbital_rotation = None
        if t1 is not None:
            final_orbital_rotation_generator = np.zeros((norb, norb), dtype=complex)
            final_orbital_rotation_generator[:nocc, nocc:] = t1
            final_orbital_rotation_generator[nocc:, :nocc] = -t1.T
            final_orbital_rotation = scipy.linalg.expm(final_orbital_rotation_generator)

        return RealUCJOperator(
            diag_coulomb_mats_alpha_alpha=diag_coulomb_mats,
            diag_coulomb_mats_alpha_beta=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=final_orbital_rotation,
        )

    def to_t_amplitudes(
        self, nocc: int | None = None
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Convert the UCJ operator to t2 (and possibly t1) amplitudes."""
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
        self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        """Apply the operator to a vector."""
        if isinstance(nelec, int):
            return NotImplemented
        if copy:
            vec = vec.copy()
        for mat, mat_alpha_beta, orbital_rotation in zip(
            self.diag_coulomb_mats_alpha_alpha,
            self.diag_coulomb_mats_alpha_beta,
            self.orbital_rotations,
        ):
            vec = apply_diag_coulomb_evolution(
                vec,
                mat=(mat, mat_alpha_beta, mat),
                time=-1.0,
                norb=norb,
                nelec=nelec,
                orbital_rotation=orbital_rotation,
                copy=False,
            )
            vec = apply_diag_coulomb_evolution(
                vec,
                mat=(-mat, -mat_alpha_beta, -mat),
                time=-1.0,
                norb=norb,
                nelec=nelec,
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
        list[tuple[int, int]],
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
    for (
        orbital_rotation,
        diag_coulomb_mat_alpha_alpha,
        diag_coulomb_mat_alpha_beta,
    ) in zip(
        orbital_rotations,
        diag_coulomb_mats_alpha_alpha,
        diag_coulomb_mats_alpha_beta,
    ):
        # orbital rotation
        n_params = norb**2
        orbital_rotation[:] = orbital_rotation_from_parameters(
            params[index : index + n_params], norb
        )
        index += n_params
        # diag coulomb matrix, alpha-alpha
        if alpha_alpha_indices:
            n_params = len(alpha_alpha_indices)
            rows, cols = zip(*alpha_alpha_indices)
            vals = params[index : index + n_params]
            diag_coulomb_mat_alpha_alpha[rows, cols] = vals
            diag_coulomb_mat_alpha_alpha[cols, rows] = vals
            index += n_params
        # diag coulomb matrix, alpha-beta
        if alpha_beta_indices:
            n_params = len(alpha_beta_indices)
            rows, cols = zip(*alpha_beta_indices)
            vals = params[index : index + n_params]
            diag_coulomb_mat_alpha_beta[rows, cols] = vals
            diag_coulomb_mat_alpha_beta[cols, rows] = vals
            index += n_params
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
        list[tuple[int, int]],
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
    for (
        orbital_rotation,
        diag_coulomb_mat_alpha_alpha,
        diag_coulomb_mat_alpha_beta,
    ) in zip(
        orbital_rotations,
        diag_coulomb_mats_alpha_alpha,
        diag_coulomb_mats_alpha_beta,
    ):
        # orbital rotation
        n_params = norb**2
        theta[index : index + n_params] = orbital_rotation_to_parameters(
            orbital_rotation
        )
        index += n_params
        # diag coulomb matrix, alpha-alpha
        if alpha_alpha_indices:
            n_params = len(alpha_alpha_indices)
            theta[index : index + n_params] = diag_coulomb_mat_alpha_alpha[
                tuple(zip(*alpha_alpha_indices))
            ]
            index += n_params
        # diag coulomb matrix, alpha-beta
        if alpha_beta_indices:
            n_params = len(alpha_beta_indices)
            theta[index : index + n_params] = diag_coulomb_mat_alpha_beta[
                tuple(zip(*alpha_beta_indices))
            ]
            index += n_params
    # final orbital rotation
    if final_orbital_rotation is not None:
        theta[index : index + norb**2] = orbital_rotation_to_parameters(
            final_orbital_rotation
        )
    return theta
