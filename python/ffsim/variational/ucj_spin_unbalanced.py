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

import jax
import jax.numpy as jnp
import numpy as np

from ffsim import gates, linalg, protocols
from ffsim.linalg.util import (
    all_indices,
    mask_from_indices,
    real_matrices_from_parameters_jax,
    real_symmetric_from_parameters_jax,
    rows_and_cols,
    unitary_from_parameters,
    unitary_from_parameters_jax,
    unitary_to_parameters,
    upper_triangular_indices,
)
from ffsim.variational.util import (
    orbital_rotation_from_t1_amplitudes,
    validate_interaction_pairs,
    validate_t2_antisymmetric,
)


@dataclass(frozen=True)
class UCJOpSpinUnbalanced(
    protocols.SupportsApplyUnitary, protocols.SupportsApproximateEquality
):
    r"""A spin-unbalanced unitary cluster Jastrow operator.

    A unitary cluster Jastrow (UCJ) operator has the form

    .. math::

        \prod_{k = 1}^L \mathcal{U}_k e^{i \mathcal{J}_k} \mathcal{U}_k^\dagger

    where each :math:`\mathcal{U_k}` is an orbital rotation and each :math:`\mathcal{J}`
    is a diagonal Coulomb operator of the form

    .. math::

        \mathcal{J} = \frac12\sum_{\substack{ij \\ \sigma \tau}}
        \mathbf{J}^{(\sigma \tau)}_{ij} n_{i\sigma} n_{j\tau}.

    Since :math:`\mathbf{J}^{(\sigma \tau)}_{ij} = \mathbf{J}^{(\tau \sigma)}_{ji}`,
    each diagonal Coulomb operator requires 3 matrices for its description:
    :math:`\mathbf{J}^{(\alpha \alpha)}`, :math:`\mathbf{J}^{(\alpha \beta)}`, and
    :math:`\mathbf{J}^{(\beta \beta)}`. The number of terms :math:`L` is referred to as
    the number of ansatz repetitions and is accessible via the ``n_reps`` attribute.

    To support variational optimization of the orbital basis, an optional final
    orbital rotation can be included in the operator, to be performed at the end.

    Args:
        validate: Whether to validate the operator attributes. Setting this to False
            skips validation, which is useful if you need to create many instances
            of this class and are confident that the attributes are valid.
        rtol: Relative numerical tolerance for validation checks.
        atol: Absolute numerical tolerance for validation checks.
    """

    diag_coulomb_mats: np.ndarray  # shape: (n_reps, 3, norb, norb)
    """The diagonal Coulomb matrices, as a Numpy array of shape
    ``(n_reps, 3, norb, norb)``. The last two axes index the rows and columns of the
    matrices, and the third from last axis, which has 3 dimensions, indexes the spin
    interaction type of the matrix: alpha-alpha, alpha-beta, and beta-beta (in that
    order). The first axis indexes the ansatz repetitions."""
    orbital_rotations: np.ndarray  # shape: (n_reps, 2, norb, norb)
    """The orbital rotations, as a Numpy array of shape ``(n_reps, 2, norb, norb)``.
    The last two axes index the rows and columns of the orbital rotations, and the third
    from last axis, which has 2 dimensions, indexes the spin sector of the orbital
    rotation: first alpha, then beta. The first axis indexes the ansatz repetitions."""
    final_orbital_rotation: np.ndarray | None = None  # shape: (2, norb, norb)
    """The optional final orbital rotation, as a Numpy array of shape
    ``(2, norb, norb)``. This can be viewed as a list of two orbital rotations, the
    first one for spin alpha and the second one for spin beta."""
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
            norb: The number of spatial orbitals.
            n_reps: The number of ansatz repetitions.
            interaction_pairs: Optional restrictions on allowed orbital interactions
                for the diagonal Coulomb operators.
                If specified, ``interaction_pairs`` should be a tuple of 3 lists,
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
            ValueError: Interaction pairs list contained out-of-range orbital indices.
            ValueError: Interaction pairs list for alpha-alpha or beta-beta interactions
                contained lower triangular pairs.
        """
        if interaction_pairs is None:
            interaction_pairs = (None, None, None)
        pairs_aa, pairs_ab, pairs_bb = interaction_pairs
        validate_interaction_pairs(pairs_aa, ordered=False, norb=norb)
        validate_interaction_pairs(pairs_ab, ordered=True, norb=norb)
        validate_interaction_pairs(pairs_bb, ordered=False, norb=norb)
        # Each same-spin diagonal Coulomb matrix has one parameter per upper triangular
        # entry unless indices are passed explicitly
        n_triu_indices = norb * (norb + 1) // 2
        n_params_aa = n_triu_indices if pairs_aa is None else len(pairs_aa)
        n_params_bb = n_triu_indices if pairs_bb is None else len(pairs_bb)
        # The different-spin diagonal Coulomb matrix has norb**2 parameters unless
        # indices are passed explicitly
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
                If specified, ``interaction_pairs`` should be a tuple of 3 lists,
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
            ValueError: Interaction pairs list contained out-of-range orbital indices.
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
        triu_indices = upper_triangular_indices(norb)
        if pairs_aa is None:
            pairs_aa = triu_indices
        if pairs_ab is None:
            pairs_ab = all_indices(norb)
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
                this_orbital_rotation[:] = unitary_from_parameters(
                    params[index : index + n_params], norb
                )
                index += n_params
            # Diag Coulomb matrices
            for indices, this_diag_coulomb_mat, symmetric in zip(
                (pairs_aa, pairs_ab, pairs_bb), diag_coulomb_mat, (True, False, True)
            ):
                n_params = len(indices)
                rows, cols = rows_and_cols(indices)
                vals = params[index : index + n_params]
                if symmetric:
                    this_diag_coulomb_mat[cols, rows] = vals
                this_diag_coulomb_mat[rows, cols] = vals
                index += n_params
        # Final orbital rotation
        final_orbital_rotation = None
        if with_final_orbital_rotation:
            final_orbital_rotation = np.zeros((2, norb, norb), dtype=complex)
            n_params = norb**2
            for this_orbital_rotation in final_orbital_rotation:
                this_orbital_rotation[:] = unitary_from_parameters(
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
            If ``interaction_pairs`` is specified, the returned parameter vector will
            incorporate only the diagonal Coulomb matrix entries corresponding to the
            specified interactions, so the original operator will not be recoverable
            from the parameter vector.

        Args:
            interaction_pairs: Optional restrictions on allowed orbital interactions
                for the diagonal Coulomb operators.
                If specified, ``interaction_pairs`` should be a tuple of 3 lists,
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
            ValueError: Interaction pairs list contained out-of-range orbital indices.
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
        triu_indices = upper_triangular_indices(norb)
        if pairs_aa is None:
            pairs_aa = triu_indices
        if pairs_ab is None:
            pairs_ab = all_indices(norb)
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
                params[index : index + n_params] = unitary_to_parameters(
                    this_orbital_rotation
                )
                index += n_params
            # Diag Coulomb matrices
            for indices, this_diag_coulomb_mat in zip(
                (pairs_aa, pairs_ab, pairs_bb),
                diag_coulomb_mat,
            ):
                n_params = len(indices)
                params[index : index + n_params] = this_diag_coulomb_mat[
                    rows_and_cols(indices)
                ]
                index += n_params
        # Final orbital rotation
        if self.final_orbital_rotation is not None:
            n_params = norb**2
            for this_orbital_rotation in self.final_orbital_rotation:
                params[index : index + n_params] = unitary_to_parameters(
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
        optimize: bool = False,
        regularization: float = 0,
        multi_stage_start: int | None = None,
        multi_stage_step: int | None = None,
        validate: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        **optimize_kwargs,
    ) -> UCJOpSpinUnbalanced:
        r"""Initialize the UCJ operator from t2 (and optionally t1) amplitudes.

        Performs a double factorization of the t2 amplitudes and constructs the
        ansatz repetitions from the terms of the decomposition, up to an optionally
        specified number of repetitions. Within the alpha-beta and same-spin terms,
        repetitions are included in decreasing order of the magnitude of the
        corresponding singular value or eigenvalue in the factorization. See the
        description of ``n_reps`` for how repetitions are divided between the two.

        Args:
            t2: The t2 amplitudes. This should be a tuple of 3 Numpy arrays,
                ``(t2aa, t2ab, t2bb)``, containing the alpha-alpha, alpha-beta, and
                beta-beta t2 amplitudes. The same-spin amplitudes ``t2aa`` and
                ``t2bb`` must be antisymmetric in the occupied indices and in the
                virtual indices.
            t1: The t1 amplitudes. This should be a pair of Numpy arrays,
                ``(t1a, t1b)``, containing the alpha and beta t1 amplitudes.
            n_reps: The number of ansatz repetitions.
                You can pass a single integer or a pair of integers.
                If a single integer, the repetitions are divided between terms
                from the alpha-beta t2 amplitudes and terms from the alpha-alpha and
                beta-beta t2 amplitudes by selecting the repetitions with the largest
                magnitudes. The magnitude of an alpha-beta repetition is the
                corresponding singular value of the alpha-beta t2 amplitudes, and the
                magnitude of a same-spin repetition is the root sum of squares of the
                corresponding eigenvalues of the alpha-alpha and beta-beta t2
                amplitudes, each scaled by :math:`1/\sqrt{2}`. With this scaling,
                the squared magnitudes of all repetitions sum to the squared norm of
                the t2 amplitudes in the spin-orbital basis, so every distinct
                excitation counts equally.
                If a pair of integers, then the first integer specifies
                the number of terms to use from the alpha-beta t2 amplitudes,
                and the second integer specifies the number of terms to use from the
                alpha-alpha and beta-beta t2 amplitudes.
                If not specified, then it is set
                to the number of terms resulting from the double factorization of the
                t2 amplitudes. If the specified number of repetitions is larger than the
                number of terms resulting from the double factorization, then the ansatz
                is padded with additional identity operators up to the specified number
                of repetitions.
            interaction_pairs: Optional restrictions on allowed orbital interactions
                for the diagonal Coulomb operators.
                If specified, ``interaction_pairs`` should be a tuple of 3 lists,
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
            optimize: Whether to optimize the tensors returned by the decomposition
                to minimize the error in the factorization.
            regularization: See
                :func:`ffsim.linalg.double_factorized_t2_compressed` for a description
                of this argument.
                This argument is ignored if ``optimize`` is set to ``False``.
            multi_stage_start: See
                :func:`ffsim.linalg.double_factorized_t2_compressed` for a description
                of this argument.
                This argument is ignored if ``optimize`` is set to ``False``.
            multi_stage_step: See
                :func:`ffsim.linalg.double_factorized_t2_compressed` for a description
                of this argument.
                This argument is ignored if ``optimize`` is set to ``False``.
            validate: Whether to validate the symmetry of the t2 amplitudes.
            rtol: Relative numerical tolerance for validation checks.
            atol: Absolute numerical tolerance for validation checks.
            optimize_kwargs: Keyword arguments to pass to
                :func:`scipy.optimize.minimize`. The default method is
                ``"L-BFGS-B"``.
                These arguments are ignored if ``optimize`` is set to ``False``.

        Returns:
            The UCJ operator with parameters initialized from the t2 amplitudes.

        Raises:
            ValueError: Interaction pairs list contained duplicate interactions.
            ValueError: Interaction pairs list contained out-of-range orbital indices.
            ValueError: Interaction pairs list for alpha-alpha or beta-beta interactions
                contained lower triangular pairs.
            ValueError: The same-spin t2 amplitudes were not antisymmetric.
        """
        if isinstance(n_reps, (int, np.integer)):
            n_reps = int(n_reps)
            if n_reps <= 0:
                raise ValueError(f"n_reps must be at least 1. Got {n_reps}.")
        if interaction_pairs is None:
            interaction_pairs = (None, None, None)
        pairs_aa, pairs_ab, pairs_bb = interaction_pairs
        t2aa, t2ab, t2bb = t2
        nocc_a, _, nvrt_a, _ = t2ab.shape
        norb = nocc_a + nvrt_a
        validate_interaction_pairs(pairs_aa, ordered=False, norb=norb)
        validate_interaction_pairs(pairs_ab, ordered=True, norb=norb)
        validate_interaction_pairs(pairs_bb, ordered=False, norb=norb)
        if validate:
            validate_t2_antisymmetric(t2aa, name="t2aa", rtol=rtol, atol=atol)
            validate_t2_antisymmetric(t2bb, name="t2bb", rtol=rtol, atol=atol)

        # The uncompressed double factorizations determine the number of terms
        # available for an integer n_reps to be divided between, and they serve as the
        # initial guess for compression
        diag_coulomb_mats_ab, orbital_rotations_ab = (
            linalg.double_factorized_t2_alpha_beta(t2ab, tol=tol)
        )
        diag_coulomb_mats_aa, orbital_rotations_aa = linalg.double_factorized_t2(
            0.5 * t2aa, tol=tol
        )
        diag_coulomb_mats_bb, orbital_rotations_bb = linalg.double_factorized_t2(
            0.5 * t2bb, tol=tol
        )

        n_reps_ab: int | None
        n_reps_same_spin: int | None
        if n_reps is None:
            n_reps_ab, n_reps_same_spin = None, None
        elif isinstance(n_reps, int):
            diag_coulomb_mats_same_spin, _ = _pair_same_spin(
                diag_coulomb_mats_aa,
                orbital_rotations_aa,
                diag_coulomb_mats_bb,
                orbital_rotations_bb,
            )
            n_reps_ab, n_reps_same_spin = _split_n_reps(
                diag_coulomb_mats_ab, diag_coulomb_mats_same_spin, n_reps
            )
        else:
            n_reps_ab, n_reps_same_spin = n_reps
            if n_reps_ab < 0:
                raise ValueError(
                    "Number of alpha-beta repetitions must be non-negative. "
                    f"Got {n_reps_ab}."
                )
            if n_reps_same_spin < 0:
                raise ValueError(
                    "Number of same-spin repetitions must be non-negative. "
                    f"Got {n_reps_same_spin}."
                )

        # A sector with no terms requested or available has nothing to optimize
        if optimize and n_reps_ab != 0 and len(orbital_rotations_ab):
            diag_coulomb_mats_ab, orbital_rotations_ab, _ = (
                linalg.double_factorized_t2_alpha_beta_compressed(
                    t2ab,
                    initial_guess=(diag_coulomb_mats_ab, orbital_rotations_ab),
                    max_terms=n_reps_ab,
                    diag_coulomb_indices=interaction_pairs,
                    regularization=regularization,
                    multi_stage_start=multi_stage_start,
                    multi_stage_step=multi_stage_step,
                    **optimize_kwargs,
                )
            )
        if optimize and n_reps_same_spin != 0 and len(orbital_rotations_aa):
            diag_coulomb_mats_aa, orbital_rotations_aa, _ = (
                linalg.double_factorized_t2_compressed(
                    0.5 * t2aa,
                    initial_guess=(diag_coulomb_mats_aa, orbital_rotations_aa),
                    max_terms=n_reps_same_spin,
                    diag_coulomb_indices=pairs_aa,
                    regularization=regularization,
                    multi_stage_start=multi_stage_start,
                    multi_stage_step=multi_stage_step,
                    **optimize_kwargs,
                )
            )
        if optimize and n_reps_same_spin != 0 and len(orbital_rotations_bb):
            diag_coulomb_mats_bb, orbital_rotations_bb, _ = (
                linalg.double_factorized_t2_compressed(
                    0.5 * t2bb,
                    initial_guess=(diag_coulomb_mats_bb, orbital_rotations_bb),
                    max_terms=n_reps_same_spin,
                    diag_coulomb_indices=pairs_bb,
                    regularization=regularization,
                    multi_stage_start=multi_stage_start,
                    multi_stage_step=multi_stage_step,
                    **optimize_kwargs,
                )
            )

        diag_coulomb_mats_same_spin, orbital_rotations_same_spin = _pair_same_spin(
            diag_coulomb_mats_aa,
            orbital_rotations_aa,
            diag_coulomb_mats_bb,
            orbital_rotations_bb,
        )

        # Truncate and pad to the requested numbers of repetitions, then concatenate
        diag_coulomb_mats = np.concatenate(
            [
                _pad_diag_coulomb_mats(diag_coulomb_mats_ab[:n_reps_ab], n_reps_ab),
                _pad_diag_coulomb_mats(
                    diag_coulomb_mats_same_spin[:n_reps_same_spin], n_reps_same_spin
                ),
            ]
        )
        orbital_rotations = np.concatenate(
            [
                _pad_orbital_rotations(orbital_rotations_ab[:n_reps_ab], n_reps_ab),
                _pad_orbital_rotations(
                    orbital_rotations_same_spin[:n_reps_same_spin], n_reps_same_spin
                ),
            ]
        )

        final_orbital_rotation = None
        if t1 is not None:
            t1a, t1b = t1
            final_orbital_rotation_a = orbital_rotation_from_t1_amplitudes(t1a)
            final_orbital_rotation_b = orbital_rotation_from_t1_amplitudes(t1b)
            final_orbital_rotation = np.stack(
                [final_orbital_rotation_a, final_orbital_rotation_b]
            )

        # Zero out diagonal coulomb matrix entries if requested
        if pairs_aa is not None:
            diag_coulomb_mats[:, 0] *= mask_from_indices(norb, pairs_aa, symmetric=True)
        if pairs_ab is not None:
            diag_coulomb_mats[:, 1] *= mask_from_indices(norb, pairs_ab)
        if pairs_bb is not None:
            diag_coulomb_mats[:, 2] *= mask_from_indices(norb, pairs_bb, symmetric=True)

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


def _pair_same_spin(
    diag_coulomb_mats_aa: np.ndarray,
    orbital_rotations_aa: np.ndarray,
    diag_coulomb_mats_bb: np.ndarray,
    orbital_rotations_bb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair up alpha-alpha and beta-beta terms, in order, into same-spin repetitions.

    If one spin sector has fewer terms, it is padded with zero diagonal Coulomb
    matrices and identity orbital rotations.
    """
    _, norb, _ = diag_coulomb_mats_aa.shape
    zero = np.zeros((norb, norb))
    eye = np.eye(norb)
    diag_coulomb_mats = np.array(
        [
            [mat_aa, zero, mat_bb]
            for mat_aa, mat_bb in itertools.zip_longest(
                diag_coulomb_mats_aa, diag_coulomb_mats_bb, fillvalue=zero
            )
        ]
    ).reshape(-1, 3, norb, norb)
    orbital_rotations = np.array(
        [
            [orb_rot_aa, orb_rot_bb]
            for orb_rot_aa, orb_rot_bb in itertools.zip_longest(
                orbital_rotations_aa, orbital_rotations_bb, fillvalue=eye
            )
        ],
        dtype=complex,
    ).reshape(-1, 2, norb, norb)
    return diag_coulomb_mats, orbital_rotations


def _split_n_reps(
    diag_coulomb_mats_ab: np.ndarray,
    diag_coulomb_mats_same_spin: np.ndarray,
    n_reps: int,
) -> tuple[int, int]:
    """Divide repetitions between alpha-beta and same-spin terms.

    The diagonal Coulomb matrices are those of the exact double factorizations. Each
    repetition is weighted by its contribution to the squared norm of the t2
    amplitudes in the spin-orbital basis, so that every distinct excitation counts
    equally. For an alpha-beta repetition this is the squared Frobenius norm of its
    diagonal Coulomb matrix in the spin-orbital basis, which is the square of a
    singular value of the alpha-beta t2 amplitudes. A same-spin diagonal Coulomb
    matrix factorizes half of the same-spin t2 amplitudes, because its alpha-alpha
    and beta-beta blocks act on each pair of orbitals in both orders, so its weight
    is twice its squared Frobenius norm. This is half the sum of the squares of an
    eigenvalue of the alpha-alpha and of the beta-beta t2 amplitudes. The ``n_reps``
    repetitions with the largest weights are selected. Any repetitions beyond the
    total number of terms are assigned to the same-spin terms, which are padded with
    identities.
    """
    weights = np.concatenate(
        [
            _spin_orbital_norms_squared(diag_coulomb_mats_ab),
            2 * _spin_orbital_norms_squared(diag_coulomb_mats_same_spin),
        ]
    )
    # Only the number of selected alpha-beta repetitions matters, so the ordering of
    # nearly equal weights within each group does not affect the result
    selected = np.argsort(-weights, kind="stable")[:n_reps]
    n_reps_ab = int(np.count_nonzero(selected < len(diag_coulomb_mats_ab)))
    return n_reps_ab, n_reps - n_reps_ab


def _spin_orbital_norms_squared(diag_coulomb_mats: np.ndarray) -> np.ndarray:
    """Squared Frobenius norms of diagonal Coulomb matrices in the spin-orbital basis.

    The alpha-beta block appears twice in the spin-orbital matrix, once above and
    once below the diagonal.
    """
    mats_aa, mats_ab, mats_bb = np.moveaxis(np.abs(diag_coulomb_mats) ** 2, -3, 0)
    return np.sum(mats_aa + 2 * mats_ab + mats_bb, axis=(-2, -1))


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


def ucj_op_spin_unbalanced_from_parameters_jax(
    params: jax.Array,
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
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """JAX version of UCJOpSpinUnbalanced.from_parameters.

    Returns:
        The diagonal Coulomb matrices, orbital rotations, and final orbital rotation
        (or None), with the same shapes as the corresponding attributes of
        UCJOpSpinUnbalanced.
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
    triu_indices = upper_triangular_indices(norb)
    if pairs_aa is None:
        pairs_aa = triu_indices
    if pairs_ab is None:
        pairs_ab = all_indices(norb)
    if pairs_bb is None:
        pairs_bb = triu_indices
    diag_coulomb_mats = jnp.zeros((n_reps, 3, norb, norb))
    orbital_rotations = jnp.zeros((n_reps, 2, norb, norb), dtype=complex)
    index = 0
    for rep in range(n_reps):
        # Orbital rotations
        n_params = norb**2
        for sigma in range(2):
            orbital_rotations = orbital_rotations.at[rep, sigma].set(
                unitary_from_parameters_jax(params[index : index + n_params], norb)
            )
            index += n_params
        # Diag Coulomb matrices
        for i, (indices, symmetric) in enumerate(
            zip((pairs_aa, pairs_ab, pairs_bb), (True, False, True))
        ):
            n_params = len(indices)
            vals = params[index : index + n_params]
            diag_coulomb_mats = diag_coulomb_mats.at[rep, i].set(
                real_symmetric_from_parameters_jax(vals, norb, indices)
                if symmetric
                else real_matrices_from_parameters_jax(vals, norb, 1, indices)[0]
            )
            index += n_params
    # Final orbital rotation
    final_orbital_rotation = None
    if with_final_orbital_rotation:
        final_orbital_rotation = jnp.zeros((2, norb, norb), dtype=complex)
        n_params = norb**2
        for sigma in range(2):
            final_orbital_rotation = final_orbital_rotation.at[sigma].set(
                unitary_from_parameters_jax(params[index : index + n_params], norb)
            )
            index += n_params
    return diag_coulomb_mats, orbital_rotations, final_orbital_rotation
