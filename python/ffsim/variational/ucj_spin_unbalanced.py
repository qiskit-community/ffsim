# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Spin-unbalanced (local) unitary cluster Jastrow ansatz."""

from __future__ import annotations

import itertools
from dataclasses import InitVar, dataclass
from typing import cast

import numpy as np
import scipy.linalg

from ffsim import gates, linalg
from ffsim.variational.util import (
    orbital_rotation_from_parameters,
    orbital_rotation_to_parameters,
)


def _validate_interaction_pairs(
    interaction_pairs: list[tuple[int, int]] | None, ordered: bool
) -> None:
    if interaction_pairs is None:
        return
    if len(set(interaction_pairs)) != len(interaction_pairs):
        raise ValueError(
            f"Duplicate interaction pairs encountered: {interaction_pairs}."
        )
    if not ordered:
        for i, j in interaction_pairs:
            if i > j:
                raise ValueError(
                    "When specifying alpha-alpha or beta-beta interaction pairs, "
                    "you must provide only upper triangular pairs. "
                    f"Got {(i, j)}, which is a lower triangular pair."
                )


@dataclass(frozen=True)
class UCJOpSpinUnbalanced:
    r"""A spin-unbalanced unitary cluster Jastrow operator.

    A unitary cluster Jastrow (UCJ) operator has the form

    .. math::

        \prod_{k = 1}^L \mathcal{U}_k e^{i \mathcal{J}_k} \mathcal{U}_k^\dagger

    where each :math:`\mathcal{U_k}` is an orbital rotation and each :math:`\mathcal{J}`
    is a diagonal Coulomb operator of the form

    .. math::

        \mathcal{J} = \frac12\sum_{\sigma \tau, ij}
        \mathbf{J}^{(\sigma \tau)}_{ij} n_{\sigma, i} n_{\tau, j}.

    Since :math:`\mathbf{J}^{(\sigma \tau)}_{ij} = \mathbf{J}^{(\tau \sigma)}_{ji}`,
    each diagonal Coulomb operator requires 3 matrices for its description:
    :math:`\mathbf{J}^{(\alpha \alpha)}`, :math:`\mathbf{J}^{(\alpha \beta)}`, and
    :math:`\mathbf{J}^{(\beta \beta)}`. The number of terms :math:`L` is referred to as
    the number of ansatz repetitions and is accessible via the `n_reps` attribute.

    To support variational optimization of the orbital basis, an optional final
    orbital rotation can be included in the operator, to be performed at the end.

    Attributes:
        diag_coulomb_mats (np.ndarray): The diagonal Coulomb matrices, as a Numpy array
            of shape `(n_reps, 3, norb, norb)`
            The last two axes index the rows and columns of
            the matrices, and the third from last axis, which has 3 dimensions, indexes
            the spin interaction type of the matrix: alpha-alpha, alpha-beta, and
            beta-beta (in that order).
            The first axis indexes the ansatz repetitions.
        orbital_rotations (np.ndarray): The orbital rotations, as a Numpy array of shape
            `(n_reps, 2, norb, norb)`. The last two axes index the rows and columns
            of the orbital rotations, and the third from last axis, which has 2
            dimensions, indexes the spin sector of the orbital rotation: first alpha,
            then beta.
            The first axis indexes the ansatz repetitions.
        final_orbital_rotation (np.ndarray | None): The optional final orbital rotation,
            as a Numpy array of shape `(2, norb, norb)`. This can be viewed as a list of
            two orbital rotations, the first one for spin alpha and the second one for
            spin beta.
    """

    diag_coulomb_mats: np.ndarray  # shape: (n_reps, 3, norb, norb)
    orbital_rotations: np.ndarray  # shape: (n_reps, 2, norb, norb)
    final_orbital_rotation: np.ndarray | None = None  # shape: (2, norb, norb)
    validate: InitVar[bool] = True
    rtol: InitVar[float] = 1e-5
    atol: InitVar[float] = 1e-8

    def __post_init__(self, validate: bool, rtol: float, atol: float):
        if validate:
            if self.diag_coulomb_mats.ndim != 4 or self.diag_coulomb_mats.shape[1] != 3:
                raise ValueError(
                    "diag_coulomb_mats should have shape (n_reps, 3, norb, norb). "
                    f"Got shape {self.diag_coulomb_mats.shape}."
                )
            if self.orbital_rotations.ndim != 4 or self.orbital_rotations.shape[1] != 2:
                raise ValueError(
                    "orbital_rotations should have shape (n_reps, 2, norb, norb). "
                    f"Got shape {self.orbital_rotations.shape}."
                )
            if (
                self.final_orbital_rotation is not None
                and self.final_orbital_rotation.ndim != 3
            ):
                raise ValueError(
                    "final_orbital_rotation should have shape (2, norb, norb). "
                    f"Got shape {self.final_orbital_rotation.shape}."
                )
            if self.diag_coulomb_mats.shape[0] != self.orbital_rotations.shape[0]:
                raise ValueError(
                    "diag_coulomb_mats and orbital_rotations should have the same "
                    "first dimension. "
                    f"Got {self.diag_coulomb_mats.shape[0]} and "
                    f"{self.orbital_rotations.shape[0]}."
                )
            if not all(
                linalg.is_real_symmetric(mats[0], rtol=rtol, atol=atol)
                and linalg.is_real_symmetric(mats[2], rtol=rtol, atol=atol)
                for mats in self.diag_coulomb_mats
            ):
                raise ValueError(
                    "alpha-alpha and beta-beta diagonal Coulomb matrices were not all "
                    "real symmetric."
                )
            if not all(
                linalg.is_unitary(orbital_rotation[0], rtol=rtol, atol=atol)
                and linalg.is_unitary(orbital_rotation[1], rtol=rtol, atol=atol)
                for orbital_rotation in self.orbital_rotations
            ):
                raise ValueError("Orbital rotations were not all unitary.")
            if self.final_orbital_rotation is not None and not (
                linalg.is_unitary(self.final_orbital_rotation[0], rtol=rtol, atol=atol)
                and linalg.is_unitary(
                    self.final_orbital_rotation[1], rtol=rtol, atol=atol
                )
            ):
                raise ValueError("Final orbital rotation was not unitary.")

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
        interaction_pairs: tuple[
            list[tuple[int, int]] | None,
            list[tuple[int, int]] | None,
            list[tuple[int, int]] | None,
        ]
        | None = None,
        with_final_orbital_rotation: bool = False,
    ) -> int:
        r"""Return the number of parameters of an ansatz with given settings.

        Args:
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

        Returns:
            The number of parameters of the ansatz.

        Raises:
            ValueError: Interaction pairs list contained duplicate interactions.
            ValueError: Interaction pairs list for alpha-alpha or beta-beta interactions
                contained lower triangular pairs.
        """
        if interaction_pairs is None:
            interaction_pairs = (None, None, None)
        pairs_aa, pairs_ab, pairs_bb = interaction_pairs
        _validate_interaction_pairs(pairs_aa, ordered=False)
        _validate_interaction_pairs(pairs_ab, ordered=True)
        _validate_interaction_pairs(pairs_bb, ordered=False)
        # Each same-spin diagonal Coulomb matrix has one parameter per upper triangular
        # entry unless indices are passed explicitly
        n_triu_indices = norb * (norb + 1) // 2
        n_params_aa = n_triu_indices if pairs_aa is None else len(pairs_aa)
        n_params_bb = n_triu_indices if pairs_bb is None else len(pairs_bb)
        # The diffent-spin diagonal Coulomb matrix has norb**2 parameters unless indices
        # are passed explicitly
        n_params_ab = norb**2 if pairs_ab is None else len(pairs_ab)
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
        interaction_pairs: tuple[
            list[tuple[int, int]] | None,
            list[tuple[int, int]] | None,
            list[tuple[int, int]] | None,
        ]
        | None = None,
        with_final_orbital_rotation: bool = False,
    ) -> UCJOpSpinUnbalanced:
        r"""Initialize the UCJ operator from a real-valued parameter vector.

        Args:
            params: The real-valued parameter vector.
            norb: The number of spatial orbitals.
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

        Returns:
            The UCJ operator constructed from the given parameters.

        Raises:
            ValueError: The number of parameters passed did not match the number
                expected based on the function inputs.
            ValueError: Interaction pairs list contained duplicate interactions.
            ValueError: Interaction pairs list for alpha-alpha or beta-beta interactions
                contained lower triangular pairs.
        """
        n_params = UCJOpSpinUnbalanced.n_params(
            norb,
            n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        if len(params) != n_params:
            raise ValueError(
                "The number of parameters passed did not match the number expected "
                "based on the function inputs. "
                f"Expected {n_params} but got {len(params)}."
            )
        if interaction_pairs is None:
            interaction_pairs = (None, None, None)
        pairs_aa, pairs_ab, pairs_bb = interaction_pairs
        mat_indices = cast(
            list[tuple[int, int]], list(itertools.product(range(norb), repeat=2))
        )
        triu_indices = cast(
            list[tuple[int, int]],
            list(itertools.combinations_with_replacement(range(norb), 2)),
        )
        if pairs_aa is None:
            pairs_aa = triu_indices
        if pairs_ab is None:
            pairs_ab = mat_indices
        if pairs_bb is None:
            pairs_bb = triu_indices
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
                (pairs_aa, pairs_ab, pairs_bb),
                diag_coulomb_mat,
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
            final_orbital_rotation = np.zeros((2, norb, norb), dtype=complex)
            n_params = norb**2
            for this_orbital_rotation in final_orbital_rotation:
                this_orbital_rotation[:] = orbital_rotation_from_parameters(
                    params[index : index + n_params], norb
                )
                index += n_params
        return UCJOpSpinUnbalanced(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=final_orbital_rotation,
        )

    def to_parameters(
        self,
        interaction_pairs: tuple[
            list[tuple[int, int]] | None,
            list[tuple[int, int]] | None,
            list[tuple[int, int]] | None,
        ]
        | None = None,
    ) -> np.ndarray:
        r"""Convert the UCJ operator to a real-valued parameter vector.

        Note:
            If `interaction_pairs` is specified, the returned parameter vector will
            incorporate only the diagonal Coulomb matrix entries corresponding to the
            specified interactions, so the original operator will not be recoverable
            from the parameter vector.

        Args:
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

        Returns:
            The real-valued parameter vector.

        Raises:
            ValueError: Interaction pairs list contained duplicate interactions.
            ValueError: Interaction pairs list for alpha-alpha or beta-beta interactions
                contained lower triangular pairs.
        """
        n_reps, _, norb, _ = self.diag_coulomb_mats.shape
        n_params = UCJOpSpinUnbalanced.n_params(
            norb,
            n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=self.final_orbital_rotation is not None,
        )

        if interaction_pairs is None:
            interaction_pairs = (None, None, None)
        pairs_aa, pairs_ab, pairs_bb = interaction_pairs
        mat_indices = cast(
            list[tuple[int, int]], list(itertools.product(range(norb), repeat=2))
        )
        triu_indices = cast(
            list[tuple[int, int]],
            list(itertools.combinations_with_replacement(range(norb), 2)),
        )
        if pairs_aa is None:
            pairs_aa = triu_indices
        if pairs_ab is None:
            pairs_ab = mat_indices
        if pairs_bb is None:
            pairs_bb = triu_indices

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
                (pairs_aa, pairs_ab, pairs_bb),
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
        n_reps: int | tuple[int, int] | None = None,
        interaction_pairs: tuple[
            list[tuple[int, int]] | None,
            list[tuple[int, int]] | None,
            list[tuple[int, int]] | None,
        ]
        | None = None,
        tol: float = 1e-8,
    ) -> UCJOpSpinUnbalanced:
        r"""Initialize the UCJ operator from t2 (and optionally t1) amplitudes.

        Performs a double-factorization of the t2 amplitudes and constructs the
        ansatz repetitions from the terms of the decomposition, up to an optionally
        specified number of repetitions. Terms are included in decreasing order
        of the magnitude of the corresponding singular value in the factorization.

        Args:
            t2: The t2 amplitudes. This should be a tuple of 3 Numpy arrays,
                `(t2aa, t2ab, t2bb)`, containing the alpha-alpha, alpha-beta, and
                beta-beta t2 amplitudes.
            t1: The t1 amplitudes. This should be a pair of Numpy arrays, `(t1a, t1b)`,
                containing the alpha and beta t1 amplitudes.
            n_reps: The number of ansatz repetitions.
                You can pass a single integer or a pair of integers.
                If a single integer, terms from the alpha-beta t2 amplitudes are
                used before including any terms from the alpha-alpha and beta-beta
                t2 amplitudes. If a pair of integers, then the first integer specifies
                the number of terms to use from the alpha-beta t2 amplitudes,
                and the second integer specifies the number of terms to use from the
                alpha-alpha and beta-beta t2 amplitudes.
                If not specified, then it is set
                to the number of terms resulting from the double-factorization of the
                t2 amplitudes. If the specified number of repetitions is larger than the
                number of terms resulting from the double-factorization, then the ansatz
                is padded with additional identity operators up to the specified number
                of repetitions.
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
            tol: Tolerance for error in the double-factorized decomposition of the
                t2 amplitudes.
                The error is defined as the maximum absolute difference between
                an element of the original tensor and the corresponding element of
                the reconstructed tensor.

        Returns:
            The UCJ operator with parameters initialized from the t2 amplitudes.

        Raises:
            ValueError: Interaction pairs list contained duplicate interactions.
            ValueError: Interaction pairs list for alpha-alpha or beta-beta interactions
                contained lower triangular pairs.
        """
        if interaction_pairs is None:
            interaction_pairs = (None, None, None)
        pairs_aa, pairs_ab, pairs_bb = interaction_pairs
        _validate_interaction_pairs(pairs_aa, ordered=False)
        _validate_interaction_pairs(pairs_bb, ordered=True)
        _validate_interaction_pairs(pairs_bb, ordered=False)

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
        if len(diag_coulomb_mats_aa) or len(diag_coulomb_mats_bb):
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
                    [orb_rot_aa, orb_rot_bb]
                    for orb_rot_aa, orb_rot_bb in itertools.zip_longest(
                        orbital_rotations_aa, orbital_rotations_bb, fillvalue=eye
                    )
                ]
            )
        else:
            diag_coulomb_mats_same_spin = np.empty((0, 3, norb, norb))
            orbital_rotations_same_spin = np.empty((0, 2, norb, norb))
        # concatenate
        if n_reps is None or isinstance(n_reps, int):
            diag_coulomb_mats = np.concatenate(
                [diag_coulomb_mats_ab, diag_coulomb_mats_same_spin]
            )[:n_reps]
            orbital_rotations = np.concatenate(
                [orbital_rotations_ab, orbital_rotations_same_spin]
            )[:n_reps]
            diag_coulomb_mats = _pad_diag_coulomb_mats(diag_coulomb_mats, n_reps)
            orbital_rotations = _pad_orbital_rotations(orbital_rotations, n_reps)
        else:
            n_reps_ab, n_reps_same_spin = n_reps
            diag_coulomb_mats_ab = _pad_diag_coulomb_mats(
                diag_coulomb_mats_ab[:n_reps_ab], n_reps_ab
            )
            orbital_rotations_ab = _pad_orbital_rotations(
                orbital_rotations_ab[:n_reps_ab], n_reps_ab
            )
            diag_coulomb_mats_same_spin = _pad_diag_coulomb_mats(
                diag_coulomb_mats_same_spin[:n_reps_same_spin], n_reps_same_spin
            )
            orbital_rotations_same_spin = _pad_orbital_rotations(
                orbital_rotations_same_spin[:n_reps_same_spin], n_reps_same_spin
            )
            diag_coulomb_mats = np.concatenate(
                [diag_coulomb_mats_ab, diag_coulomb_mats_same_spin]
            )
            orbital_rotations = np.concatenate(
                [orbital_rotations_ab, orbital_rotations_same_spin]
            )

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
        if pairs_bb is not None:
            mask = np.zeros((norb, norb), dtype=bool)
            rows, cols = zip(*pairs_bb)
            mask[rows, cols] = True
            mask[cols, rows] = True
            diag_coulomb_mats[:, 2] *= mask

        return UCJOpSpinUnbalanced(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=final_orbital_rotation,
        )

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        if isinstance(nelec, int):
            return NotImplemented
        if copy:
            vec = vec.copy()
        eye = np.eye(norb)
        current_basis = np.stack([eye, eye])
        for diag_coulomb_mat, orbital_rotation in zip(
            self.diag_coulomb_mats, self.orbital_rotations
        ):
            vec = gates.apply_orbital_rotation(
                vec,
                orbital_rotation.transpose(0, 2, 1).conj() @ current_basis,
                norb=norb,
                nelec=nelec,
                copy=False,
            )
            vec = gates.apply_diag_coulomb_evolution(
                vec,
                diag_coulomb_mat,
                time=-1.0,
                norb=norb,
                nelec=nelec,
                copy=False,
            )
            current_basis = orbital_rotation
        if self.final_orbital_rotation is None:
            vec = gates.apply_orbital_rotation(
                vec, current_basis, norb=norb, nelec=nelec, copy=False
            )
        else:
            vec = gates.apply_orbital_rotation(
                vec,
                self.final_orbital_rotation @ current_basis,
                norb=norb,
                nelec=nelec,
                copy=False,
            )
        return vec

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        if isinstance(other, UCJOpSpinUnbalanced):
            if not np.allclose(
                self.diag_coulomb_mats, other.diag_coulomb_mats, rtol=rtol, atol=atol
            ):
                return False
            if not np.allclose(
                self.orbital_rotations, other.orbital_rotations, rtol=rtol, atol=atol
            ):
                return False
            if (self.final_orbital_rotation is None) != (
                other.final_orbital_rotation is None
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


def _pad_diag_coulomb_mats(diag_coulomb_mats: np.ndarray, n_reps: int | None):
    n_vecs, _, norb, _ = diag_coulomb_mats.shape
    if n_reps is not None and n_vecs < n_reps:
        diag_coulomb_mats = np.concatenate(
            [
                diag_coulomb_mats,
                np.zeros((n_reps - n_vecs, 3, norb, norb)),
            ]
        )
    return diag_coulomb_mats


def _pad_orbital_rotations(orbital_rotations: np.ndarray, n_reps: int | None):
    n_vecs, _, norb, _ = orbital_rotations.shape
    if n_reps is not None and n_vecs < n_reps:
        eye = np.eye(norb)
        orbital_rotations = np.concatenate(
            [
                orbital_rotations,
                np.stack([np.stack([eye, eye]) for _ in range(n_reps - n_vecs)]),
            ]
        )
    return orbital_rotations
