# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Closed-shell (local) unitary cluster Jastrow ansatz."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Tuple, cast

import numpy as np
import scipy.linalg

from ffsim import gates, linalg
from ffsim.variational.util import (
    orbital_rotation_from_parameters,
    orbital_rotation_to_parameters,
)


def _validate_diag_coulomb_indices(indices: list[tuple[int, int]] | None):
    if indices is not None:
        for i, j in indices:
            if i > j:
                raise ValueError(
                    "When specifying alpha-alpha or alpha-beta diagonal Coulomb "
                    "indices, you must provide only upper trianglular indices. "
                    f"Got {(i, j)}, which is a lower triangular index."
                )


@dataclass(frozen=True)
class UCJOperatorClosedShell:
    r"""An open-shell unitary cluster Jastrow operator.

    A unitary cluster Jastrow (UCJ) operator has the form

    .. math::

        \prod_{k = 1}^L \mathcal{U}_k e^{i \mathcal{J}_k} \mathcal{U}_k^\dagger

    where each :math:`\mathcal{U_k}` is an orbital rotation and each :math:`\mathcal{J}`
    is a diagonal Coulomb operator of the form

    .. math::

        \mathcal{J} = \frac12\sum_{\sigma \tau, ij}
        \mathbf{J}^{(\sigma \tau)}_{ij} n_{\sigma, i} n_{\tau, j}.

    For this closed-shell operator, we require that
    :math:`\mathbf{J}^{(\alpha \alpha)} = \mathbf{J}^{(\beta \beta)}` and
    :math:`\mathbf{J}^{(\alpha \beta)} = \mathbf{J}^{(\beta \alpha)}`.
    Therefore, each diagonal Coulomb operator is described by 2 matrices,
    :math:`\mathbf{J}^{(\alpha \alpha)}` and :math:`\mathbf{J}^{(\alpha \beta)}`, and
    both of these matrices are symmetric.
    Furthermore, each orbital rotation is described by a single matrix because the
    same orbital rotation is applied to both spin alpha and spin beta.
    The number of terms :math:`L` is referred to as the
    number of ansatz repetitions and is accessible via the `n_reps` attribute.

    To support variational optimization of the orbital basis, an optional final
    orbital rotation can be included in the operator, to be performed at the end.

    Attributes:
        diag_coulomb_mats (np.ndarray): The diagonal Coulomb matrices, as a Numpy array
            of shape `(n_reps, 2, norb, norb)`
            The last two axes index the rows and columns of
            the matrices, and the third from last axis, which has 2 dimensions, indexes
            the spin interaction type of the matrix: alpha-alpha, and then alpha-beta.
            The first axis indexes the ansatz repetitions.
        orbital_rotations (np.ndarray): The orbital rotations, as a Numpy array
            of shape `(n_reps, norb, norb)`.
        final_orbital_rotation (np.ndarray): The optional final orbital rotation, as
            a Numpy array of shape `(norb, norb)`.
    """

    diag_coulomb_mats: np.ndarray  # shape: (n_reps, 2, norb, norb)
    orbital_rotations: np.ndarray  # shape: (n_reps, norb, norb)
    final_orbital_rotation: np.ndarray | None = None  # shape: (norb, norb)

    @property
    def norb(self):
        """The number of spatial orbitals."""
        return self.diag_coulomb_mats.shape[-1]

    @property
    def n_reps(self):
        """The number of ansatz repetitions."""
        return self.diag_coulomb_mats.shape[0]

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
        # Each diagonal Coulomb matrix has one parameter per upper triangular
        # entry unless indices are passed explicitly
        n_triu_indices = norb * (norb + 1) // 2
        n_params_aa = (
            n_triu_indices if alpha_alpha_indices is None else len(alpha_alpha_indices)
        )
        n_params_ab = (
            n_triu_indices if alpha_beta_indices is None else len(alpha_beta_indices)
        )
        # Each orbital rotation has norb**2 parameters
        return (
            n_reps * (n_params_aa + n_params_ab + norb**2)
            + with_final_orbital_rotation * norb**2
        )

    @staticmethod
    def from_parameters(
        params: np.ndarray,
        *,
        norb: int,
        n_reps: int,
        alpha_alpha_indices: list[tuple[int, int]] | None = None,
        alpha_beta_indices: list[tuple[int, int]] | None = None,
        with_final_orbital_rotation: bool = False,
    ) -> UCJOperatorClosedShell:
        r"""Initialize the UCJ operator from a real-valued parameter vector.

        Args:
            params: The real-valued parameter vector.
            norb: The number of spatial orbitals.
            n_reps: The number of ansatz repetitions.
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
            ValueError: The number of parameters passed did not match the number
                expected based on the function inputs.
            ValueError: alpha_alpha_indices contains lower triangular indices.
            ValueError: alpha_beta_indices contains lower triangular indices.
        """
        _validate_diag_coulomb_indices(alpha_alpha_indices)
        _validate_diag_coulomb_indices(alpha_beta_indices)
        n_params = UCJOperatorClosedShell.n_params(
            norb,
            n_reps,
            alpha_alpha_indices=alpha_alpha_indices,
            alpha_beta_indices=alpha_beta_indices,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        if len(params) != n_params:
            raise ValueError(
                "The number of parameters passed did not match the number expected "
                "based on the function inputs. "
                f"Expected {n_params} but got {len(params)}."
            )
        triu_indices = cast(
            List[Tuple[int, int]],
            list(itertools.combinations_with_replacement(range(norb), 2)),
        )
        if alpha_alpha_indices is None:
            alpha_alpha_indices = triu_indices
        if alpha_beta_indices is None:
            alpha_beta_indices = triu_indices
        diag_coulomb_mats = np.zeros((n_reps, 2, norb, norb))
        orbital_rotations = np.zeros((n_reps, norb, norb), dtype=complex)
        index = 0
        for orbital_rotation, diag_coulomb_mat in zip(
            orbital_rotations, diag_coulomb_mats
        ):
            # Orbital rotations
            n_params = norb**2
            orbital_rotation[:] = orbital_rotation_from_parameters(
                params[index : index + n_params], norb
            )
            index += n_params
            # Diag Coulomb matrices
            for indices, this_diag_coulomb_mat in zip(
                (alpha_alpha_indices, alpha_beta_indices), diag_coulomb_mat
            ):
                if indices:
                    n_params = len(indices)
                    rows, cols = zip(*indices)
                    vals = params[index : index + n_params]
                    this_diag_coulomb_mat[cols, rows] = vals
                    this_diag_coulomb_mat[rows, cols] = vals
                    index += n_params
        # Final orbital rotation
        final_orbital_rotation = None
        if with_final_orbital_rotation:
            final_orbital_rotation = orbital_rotation_from_parameters(
                params[index:], norb
            )
        return UCJOperatorClosedShell(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=final_orbital_rotation,
        )

    def to_parameters(
        self,
        *,
        alpha_alpha_indices: list[tuple[int, int]] | None = None,
        alpha_beta_indices: list[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        r"""Convert the UCJ operator to a real-valued parameter vector.

        If `alpha_alpha_indices` or `alpha_beta_indices` is
        specified, the returned parameter vector will incorporate only the diagonal
        Coulomb matrix entries corresponding to the given indices, so the original
        operator will not be recoverable from the parameter vector.

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
        n_reps, _, norb, _ = self.diag_coulomb_mats.shape
        triu_indices = cast(
            List[Tuple[int, int]],
            list(itertools.combinations_with_replacement(range(norb), 2)),
        )
        if alpha_alpha_indices is None:
            alpha_alpha_indices = triu_indices
        if alpha_beta_indices is None:
            alpha_beta_indices = triu_indices
        n_params = UCJOperatorClosedShell.n_params(
            norb,
            n_reps,
            alpha_alpha_indices=alpha_alpha_indices,
            alpha_beta_indices=alpha_beta_indices,
            with_final_orbital_rotation=self.final_orbital_rotation is not None,
        )
        params = np.zeros(n_params)
        index = 0
        for orbital_rotation, diag_coulomb_mat in zip(
            self.orbital_rotations, self.diag_coulomb_mats
        ):
            # Orbital rotations
            n_params = norb**2
            params[index : index + n_params] = orbital_rotation_to_parameters(
                orbital_rotation
            )
            index += n_params
            # Diag Coulomb matrices
            for indices, this_diag_coulomb_mat in zip(
                (alpha_alpha_indices, alpha_beta_indices), diag_coulomb_mat
            ):
                if indices:
                    n_params = len(indices)
                    params[index : index + n_params] = this_diag_coulomb_mat[
                        tuple(zip(*indices))
                    ]
                    index += n_params
        # Final orbital rotation
        if self.final_orbital_rotation is not None:
            params[index:] = orbital_rotation_to_parameters(self.final_orbital_rotation)
        return params

    @staticmethod
    def from_t_amplitudes(
        t2: np.ndarray,
        *,
        t1: np.ndarray | None = None,
        n_reps: int | None = None,
        alpha_alpha_indices: list[tuple[int, int]] | None = None,
        alpha_beta_indices: list[tuple[int, int]] | None = None,
        tol: float = 1e-8,
    ) -> UCJOperatorClosedShell:
        """Initialize the UCJ operator from t2 (and optionally t1) amplitudes.

        Performs a double-factorization of the t2 amplitudes and constructs the
        ansatz repetitions from the terms of the decomposition, up to an optionally
        specified number of ansatz repetitions. Terms are included in decreasing order
        of the absolute value of the corresponding eigenvalue in the factorization.

        Args:
            t2: The t2 amplitudes.
            t1: The t1 amplitudes.
            n_reps: The number of ansatz repetitions.
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
            tol: Tolerance for error in the double-factorized decomposition of the
                t2 amplitudes.
                The error is defined as the maximum absolute difference between
                an element of the original tensor and the corresponding element of
                the reconstructed tensor.
        """
        nocc, _, nvrt, _ = t2.shape
        norb = nocc + nvrt

        diag_coulomb_mats, orbital_rotations = linalg.double_factorized_t2(t2, tol=tol)
        diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)[:n_reps]
        diag_coulomb_mats = np.stack([diag_coulomb_mats, diag_coulomb_mats], axis=1)
        orbital_rotations = orbital_rotations.reshape(-1, norb, norb)[:n_reps]

        final_orbital_rotation = None
        if t1 is not None:
            final_orbital_rotation_generator = np.zeros((norb, norb), dtype=complex)
            final_orbital_rotation_generator[:nocc, nocc:] = t1
            final_orbital_rotation_generator[nocc:, :nocc] = -t1.T
            final_orbital_rotation = scipy.linalg.expm(final_orbital_rotation_generator)

        # Zero out diagonal coulomb matrix entries if requested
        if alpha_alpha_indices is not None:
            mask = np.zeros((norb, norb), dtype=bool)
            rows, cols = zip(*alpha_alpha_indices)
            mask[rows, cols] = True
            mask[cols, rows] = True
            diag_coulomb_mats[:, 0] *= mask
        if alpha_beta_indices is not None:
            mask = np.zeros((norb, norb), dtype=bool)
            rows, cols = zip(*alpha_beta_indices)
            mask[rows, cols] = True
            mask[cols, rows] = True
            diag_coulomb_mats[:, 1] *= mask

        return UCJOperatorClosedShell(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=final_orbital_rotation,
        )

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: tuple[int, int], copy: bool
    ) -> np.ndarray:
        if copy:
            vec = vec.copy()
        for (diag_coulomb_mat_aa, diag_coulomb_mat_ab), orbital_rotation in zip(
            self.diag_coulomb_mats, self.orbital_rotations
        ):
            vec = gates.apply_diag_coulomb_evolution(
                vec,
                (diag_coulomb_mat_aa, diag_coulomb_mat_ab, diag_coulomb_mat_aa),
                time=-1.0,
                norb=norb,
                nelec=nelec,
                orbital_rotation=orbital_rotation,
                copy=False,
            )
        if self.final_orbital_rotation is not None:
            vec = gates.apply_orbital_rotation(
                vec,
                mat=self.final_orbital_rotation,
                norb=norb,
                nelec=nelec,
                copy=False,
            )
        return vec

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        if isinstance(other, UCJOperatorClosedShell):
            if not np.allclose(
                self.diag_coulomb_mats, other.diag_coulomb_mats, rtol=rtol, atol=atol
            ):
                return False
            if not np.allclose(
                self.orbital_rotations, other.orbital_rotations, rtol=rtol, atol=atol
            ):
                return False
            if (
                self.final_orbital_rotation is None
                and other.final_orbital_rotation is not None
            ) or (
                self.final_orbital_rotation is not None
                and other.final_orbital_rotation is None
            ):
                return False
            if self.final_orbital_rotation is not None:
                return np.allclose(
                    cast(np.ndarray, self.final_orbital_rotation),
                    cast(np.ndarray, other.final_orbital_rotation),
                    rtol=rtol,
                    atol=atol,
                )
            return True
        return NotImplemented
