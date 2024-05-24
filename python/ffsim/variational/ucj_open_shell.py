# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Open-shell (local) unitary cluster Jastrow ansatz."""

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
                    "When specifying alpha-alpha or beta-beta diagonal Coulomb "
                    "indices, you must provide only upper trianglular indices. "
                    f"Got {(i, j)}, which is a lower triangular index."
                )


@dataclass(frozen=True)
class UCJOperatorOpenShell:
    r"""An open-shell unitary cluster Jastrow operator.

    TODO

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
        diag_coulomb_mats (np.ndarray): The diagonal Coulomb matrices.
        orbital_rotations (np.ndarray): The orbital rotations.
        final_orbital_rotation (np.ndarray): The optional final orbital rotation.
    """

    diag_coulomb_mats: np.ndarray  # shape: (n_reps, 3, norb, norb)
    orbital_rotations: np.ndarray  # shape: (n_reps, 2, norb, norb)
    final_orbital_rotation: np.ndarray | None = None  # shape: (2, norb, norb)

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
        beta_beta_indices: list[tuple[int, int]] | None = None,
        with_final_orbital_rotation: bool = False,
    ) -> int:
        """Return the number of parameters of an ansatz with given settings."""
        _validate_diag_coulomb_indices(alpha_alpha_indices)
        _validate_diag_coulomb_indices(beta_beta_indices)
        # Each same-spin diagonal Coulomb matrix has one parameter per upper triangular
        # entry unless indices are passed explicitly
        n_triu_indices = norb * (norb + 1) // 2
        n_params_aa = (
            n_triu_indices if alpha_alpha_indices is None else len(alpha_alpha_indices)
        )
        n_params_bb = (
            n_triu_indices if beta_beta_indices is None else len(beta_beta_indices)
        )
        # The diffent-spin diagonal Coulomb matrix has norb**2 parameters unless indices
        # are passed explicitly
        n_params_ab = norb**2 if alpha_beta_indices is None else len(alpha_beta_indices)
        # Each orbital rotation has norb**2 parameters per spin
        return (
            n_reps * (n_params_aa + n_params_ab + n_params_bb + 2 * norb**2)
            + with_final_orbital_rotation * 2 * norb**2
        )

    @staticmethod
    def from_parameters(
        params: np.ndarray,
        *,
        norb: int,
        n_reps: int,
        alpha_alpha_indices: list[tuple[int, int]] | None = None,
        alpha_beta_indices: list[tuple[int, int]] | None = None,
        beta_beta_indices: list[tuple[int, int]] | None = None,
        with_final_orbital_rotation: bool = False,
    ) -> UCJOperatorOpenShell:
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
        _validate_diag_coulomb_indices(beta_beta_indices)
        n_params = UCJOperatorOpenShell.n_params(
            norb,
            n_reps,
            alpha_alpha_indices=alpha_alpha_indices,
            alpha_beta_indices=alpha_beta_indices,
            beta_beta_indices=beta_beta_indices,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        if len(params) != n_params:
            raise ValueError(
                "The number of parameters passed did not match the expected number. "
                f"Expected {n_params} but got {len(params)}."
            )
        mat_indices = cast(
            List[Tuple[int, int]], list(itertools.product(range(norb), repeat=2))
        )
        triu_indices = cast(
            List[Tuple[int, int]],
            list(itertools.combinations_with_replacement(range(norb), 2)),
        )
        if alpha_alpha_indices is None:
            alpha_alpha_indices = triu_indices
        if alpha_beta_indices is None:
            alpha_beta_indices = mat_indices
        if beta_beta_indices is None:
            beta_beta_indices = triu_indices
        diag_coulomb_mats = np.zeros((n_reps, 3, norb, norb))
        orbital_rotations = np.zeros((n_reps, 2, norb, norb), dtype=complex)
        index = 0
        for orbital_rotation, diag_coulomb_mat in zip(
            orbital_rotations, diag_coulomb_mats
        ):
            # Orbital rotations
            n_params = norb**2
            for this_orbital_rotation in orbital_rotation:
                this_orbital_rotation[:] = orbital_rotation_from_parameters(
                    params[index : index + n_params], norb
                )
                index += n_params
            # Diag Coulomb matrices
            for indices, this_diag_coulomb_mat in zip(
                (alpha_alpha_indices, alpha_beta_indices, beta_beta_indices),
                diag_coulomb_mat,
            ):
                if indices:
                    n_params = len(indices)
                    rows, cols = zip(*indices)
                    vals = params[index : index + n_params]
                    this_diag_coulomb_mat[rows, cols] = vals
                    this_diag_coulomb_mat[cols, rows] = vals
                    index += n_params
        # Final orbital rotation
        final_orbital_rotation = None
        if with_final_orbital_rotation:
            final_orbital_rotation = np.zeros((2, norb, norb), dtype=complex)
            n_params = norb**2
            for this_orbital_rotation in final_orbital_rotation:
                this_orbital_rotation[:] = orbital_rotation_from_parameters(
                    params[index : index + n_params], norb
                )
                index += n_params
        return UCJOperatorOpenShell(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=final_orbital_rotation,
        )

    def to_parameters(
        self,
        *,
        alpha_alpha_indices: list[tuple[int, int]] | None = None,
        alpha_beta_indices: list[tuple[int, int]] | None = None,
        beta_beta_indices: list[tuple[int, int]] | None = None,
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
        _validate_diag_coulomb_indices(beta_beta_indices)
        n_reps, _, norb, _ = self.diag_coulomb_mats.shape
        mat_indices = cast(
            List[Tuple[int, int]], list(itertools.product(range(norb), repeat=2))
        )
        triu_indices = cast(
            List[Tuple[int, int]],
            list(itertools.combinations_with_replacement(range(norb), 2)),
        )
        if alpha_alpha_indices is None:
            alpha_alpha_indices = triu_indices
        if alpha_beta_indices is None:
            alpha_beta_indices = mat_indices
        if beta_beta_indices is None:
            beta_beta_indices = triu_indices
        n_params = UCJOperatorOpenShell.n_params(
            norb,
            n_reps,
            alpha_alpha_indices=alpha_alpha_indices,
            alpha_beta_indices=alpha_beta_indices,
            beta_beta_indices=beta_beta_indices,
            with_final_orbital_rotation=self.final_orbital_rotation is not None,
        )
        params = np.zeros(n_params)
        index = 0
        for orbital_rotation, diag_coulomb_mat in zip(
            self.orbital_rotations, self.diag_coulomb_mats
        ):
            # Orbital rotations
            n_params = norb**2
            for this_orbital_rotation in orbital_rotation:
                params[index : index + n_params] = orbital_rotation_to_parameters(
                    this_orbital_rotation
                )
                index += n_params
            # Diag Coulomb matrices
            for indices, this_diag_coulomb_mat in zip(
                (alpha_alpha_indices, alpha_beta_indices, beta_beta_indices),
                diag_coulomb_mat,
            ):
                if indices:
                    n_params = len(indices)
                    params[index : index + n_params] = this_diag_coulomb_mat[
                        tuple(zip(*indices))
                    ]
                    index += n_params
        # Final orbital rotation
        if self.final_orbital_rotation is not None:
            n_params = norb**2
            for this_orbital_rotation in self.final_orbital_rotation:
                params[index : index + n_params] = orbital_rotation_to_parameters(
                    this_orbital_rotation
                )
                index += n_params
        return params

    @staticmethod
    def from_t_amplitudes(
        t2: tuple[np.ndarray, np.ndarray, np.ndarray],
        *,
        t1: tuple[np.ndarray, np.ndarray] | None = None,
        n_reps: int | None = None,
        alpha_alpha_indices: list[tuple[int, int]] | None = None,
        alpha_beta_indices: list[tuple[int, int]] | None = None,
        beta_beta_indices: list[tuple[int, int]] | None = None,
        tol: float = 1e-8,
    ) -> UCJOperatorOpenShell:
        """Initialize the UCJ operator from t2 (and optionally t1) amplitudes."""
        t2aa, t2ab, t2bb = t2
        nocc_a, nocc_b, nvrt_a, _ = t2ab.shape
        norb = nocc_a + nvrt_a

        # alpha-beta
        diag_coulomb_mats_ab, orbital_rotations_ab = (
            linalg.double_factorized_t2_alpha_beta(t2ab, tol=tol)
        )
        diag_coulomb_mats_ab = diag_coulomb_mats_ab.reshape(-1, 3, norb, norb)
        orbital_rotations_ab = orbital_rotations_ab.reshape(-1, 2, norb, norb)
        # alpha-alpha and beta-beta
        diag_coulomb_mats_aa, orbital_rotations_aa = linalg.double_factorized_t2(
            t2aa, tol=tol
        )
        diag_coulomb_mats_bb, orbital_rotations_bb = linalg.double_factorized_t2(
            t2bb, tol=tol
        )
        diag_coulomb_mats_aa = diag_coulomb_mats_aa.reshape(-1, norb, norb)
        orbital_rotations_aa = orbital_rotations_aa.reshape(-1, norb, norb)
        diag_coulomb_mats_bb = diag_coulomb_mats_bb.reshape(-1, norb, norb)
        orbital_rotations_bb = orbital_rotations_bb.reshape(-1, norb, norb)
        zero = np.zeros((norb, norb))
        diag_coulomb_mats_same_spin = np.stack(
            [
                [mat_aa, zero, mat_bb]
                for mat_aa, mat_bb in itertools.zip_longest(
                    diag_coulomb_mats_aa, diag_coulomb_mats_bb, fillvalue=zero
                )
            ]
        )
        eye = np.eye(norb)
        orbital_rotations_same_spin = np.stack(
            [
                [orbital_rotation_aa, orbital_rotation_bb]
                for orbital_rotation_aa, orbital_rotation_bb in itertools.zip_longest(
                    orbital_rotations_aa, orbital_rotations_bb, fillvalue=eye
                )
            ]
        )
        # concatenate
        # TODO might need to scale by a factor of 2
        diag_coulomb_mats = np.concatenate(
            [diag_coulomb_mats_ab, diag_coulomb_mats_same_spin]
        )[:n_reps]
        orbital_rotations = np.concatenate(
            [orbital_rotations_ab, orbital_rotations_same_spin]
        )[:n_reps]

        final_orbital_rotation = None
        if t1 is not None:
            t1a, t1b = t1

            final_orbital_rotation_generator_a = np.zeros((norb, norb), dtype=complex)
            final_orbital_rotation_generator_a[:nocc_a, nocc_a:] = t1a
            final_orbital_rotation_generator_a[nocc_a:, :nocc_a] = -t1a.T
            final_orbital_rotation_a = scipy.linalg.expm(
                final_orbital_rotation_generator_a
            )

            final_orbital_rotation_generator_b = np.zeros((norb, norb), dtype=complex)
            final_orbital_rotation_generator_b[:nocc_b, nocc_b:] = t1b
            final_orbital_rotation_generator_b[nocc_b:, :nocc_b] = -t1b.T
            final_orbital_rotation_b = scipy.linalg.expm(
                final_orbital_rotation_generator_b
            )
            final_orbital_rotation = np.stack(
                [final_orbital_rotation_a, final_orbital_rotation_b]
            )

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
        if beta_beta_indices is not None:
            mask = np.zeros((norb, norb), dtype=bool)
            rows, cols = zip(*beta_beta_indices)
            mask[rows, cols] = True
            mask[cols, rows] = True
            diag_coulomb_mats[:, 2] *= mask

        return UCJOperatorOpenShell(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=final_orbital_rotation,
        )

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: tuple[int, int], copy: bool
    ) -> np.ndarray:
        """Apply the operator to a vector."""
        if copy:
            vec = vec.copy()
        for diag_coulomb_mat, orbital_rotation in zip(
            self.diag_coulomb_mats, self.orbital_rotations
        ):
            vec = gates.apply_diag_coulomb_evolution(
                vec,
                diag_coulomb_mat,
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
        if isinstance(other, UCJOperatorOpenShell):
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
