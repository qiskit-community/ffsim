# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Spinless (local) unitary cluster Jastrow ansatz."""

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
                    "When specifying interaction pairs, "
                    "you must provide only upper triangular pairs. "
                    f"Got {(i, j)}, which is a lower triangular pair."
                )


@dataclass(frozen=True)
class UCJOpSpinless:
    r"""A spinless unitary cluster Jastrow operator.

    A spinless unitary cluster Jastrow (UCJ) operator has the form

    .. math::

        \prod_{k = 1}^L \mathcal{U}_k e^{i \mathcal{J}_k} \mathcal{U}_k^\dagger

    where each :math:`\mathcal{U_k}` is an orbital rotation and each :math:`\mathcal{J}`
    is a diagonal Coulomb operator of the form

    .. math::

        \mathcal{J} = \frac12\sum_{ij}
        \mathbf{J}_{ij} n_{i} n_{j}.

    where \mathbf{J} is a real symmetric matrix.
    The number of terms :math:`L` is referred to as the
    number of ansatz repetitions and is accessible via the `n_reps` attribute.

    To support variational optimization of the orbital basis, an optional final
    orbital rotation can be included in the operator, to be performed at the end.

    Attributes:
        diag_coulomb_mats (np.ndarray): The diagonal Coulomb matrices, as a Numpy array
            of shape `(n_reps, norb, norb)`
        orbital_rotations (np.ndarray): The orbital rotations, as a Numpy array
            of shape `(n_reps, norb, norb)`.
        final_orbital_rotation (np.ndarray | None): The optional final orbital rotation,
            as a Numpy array of shape `(norb, norb)`.
    """

    diag_coulomb_mats: np.ndarray  # shape: (n_reps, norb, norb)
    orbital_rotations: np.ndarray  # shape: (n_reps, norb, norb)
    final_orbital_rotation: np.ndarray | None = None  # shape: (norb, norb)
    validate: InitVar[bool] = True
    rtol: InitVar[float] = 1e-5
    atol: InitVar[float] = 1e-8

    def __post_init__(self, validate: bool, rtol: float, atol: float):
        if validate:
            if self.diag_coulomb_mats.ndim != 3:
                raise ValueError(
                    "diag_coulomb_mats should have shape (n_reps, norb, norb). "
                    f"Got shape {self.diag_coulomb_mats.shape}."
                )
            if self.orbital_rotations.ndim != 3:
                raise ValueError(
                    "orbital_rotations should have shape (n_reps, norb, norb). "
                    f"Got shape {self.orbital_rotations.shape}."
                )
            if (
                self.final_orbital_rotation is not None
                and self.final_orbital_rotation.ndim != 2
            ):
                raise ValueError(
                    "final_orbital_rotation should have shape (norb, norb). "
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
                linalg.is_real_symmetric(mat, rtol=rtol, atol=atol)
                for mat in self.diag_coulomb_mats
            ):
                raise ValueError(
                    "Diagonal Coulomb matrices were not all real symmetric."
                )
            if not all(
                linalg.is_unitary(orbital_rotation, rtol=rtol, atol=atol)
                for orbital_rotation in self.orbital_rotations
            ):
                raise ValueError("Orbital rotations were not all unitary.")
            if self.final_orbital_rotation is not None and not linalg.is_unitary(
                self.final_orbital_rotation, rtol=rtol, atol=atol
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
        interaction_pairs: list[tuple[int, int]] | None = None,
        with_final_orbital_rotation: bool = False,
    ) -> int:
        r"""Return the number of parameters of an ansatz with given settings.

        Args:
            n_reps: The number of ansatz repetitions.
            interaction_pairs: Optional restrictions on allowed orbital interactions
                for the diagonal Coulomb operators.
                If specified, `interaction_pairs` should be a list of integer pairs
                representing the orbitals that are allowed to interact. These pairs
                can also be interpreted as indices of diagonal Coulomb matrix entries
                that are allowed to be nonzero.
                Each integer pair must be upper triangular, that is, of the form
                :math:`(i, j)` where :math:`i \leq j`.
            with_final_orbital_rotation: Whether to include a final orbital rotation
                in the operator.

        Returns:
            The number of parameters of the ansatz.

        Raises:
            ValueError: Interaction pairs list contained duplicate interactions.
            ValueError: Interaction pairs list contained lower triangular pairs.
        """
        _validate_interaction_pairs(interaction_pairs, ordered=False)
        # Each diagonal Coulomb matrix has one parameter per upper triangular
        # entry unless indices are passed explicitly
        n_triu_indices = norb * (norb + 1) // 2
        n_params_diag_coulomb = (
            n_triu_indices if interaction_pairs is None else len(interaction_pairs)
        )
        # Each orbital rotation has norb**2 parameters
        return (
            n_reps * (n_params_diag_coulomb + norb**2)
            + with_final_orbital_rotation * norb**2
        )

    @staticmethod
    def from_parameters(
        params: np.ndarray,
        *,
        norb: int,
        n_reps: int,
        interaction_pairs: list[tuple[int, int]] | None = None,
        with_final_orbital_rotation: bool = False,
    ) -> UCJOpSpinless:
        r"""Initialize the UCJ operator from a real-valued parameter vector.

        Args:
            params: The real-valued parameter vector.
            norb: The number of spatial orbitals.
            n_reps: The number of ansatz repetitions.
            interaction_pairs: Optional restrictions on allowed orbital interactions
                for the diagonal Coulomb operators.
                If specified, `interaction_pairs` should be a list of integer pairs
                representing the orbitals that are allowed to interact. These pairs
                can also be interpreted as indices of diagonal Coulomb matrix entries
                that are allowed to be nonzero.
                Each integer pair must be upper triangular, that is, of the form
                :math:`(i, j)` where :math:`i \leq j`.
            with_final_orbital_rotation: Whether to include a final orbital rotation
                in the operator.

        Returns:
            The UCJ operator constructed from the given parameters.

        Raises:
            ValueError: The number of parameters passed did not match the number
                expected based on the function inputs.
            ValueError: Interaction pairs list contained duplicate interactions.
            ValueError: Interaction pairs list contained lower triangular pairs.
        """
        n_params = UCJOpSpinless.n_params(
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
        triu_indices = cast(
            list[tuple[int, int]],
            list(itertools.combinations_with_replacement(range(norb), 2)),
        )
        if interaction_pairs is None:
            interaction_pairs = triu_indices
        diag_coulomb_mats = np.zeros((n_reps, norb, norb))
        orbital_rotations = np.zeros((n_reps, norb, norb), dtype=complex)
        index = 0
        for orbital_rotation, diag_coulomb_mat in zip(
            orbital_rotations, diag_coulomb_mats
        ):
            # Orbital rotation
            n_params = norb**2
            orbital_rotation[:] = orbital_rotation_from_parameters(
                params[index : index + n_params], norb
            )
            index += n_params
            # Diag Coulomb matrix
            if interaction_pairs:
                n_params = len(interaction_pairs)
                rows, cols = zip(*interaction_pairs)
                vals = params[index : index + n_params]
                diag_coulomb_mat[cols, rows] = vals
                diag_coulomb_mat[rows, cols] = vals
                index += n_params
        # Final orbital rotation
        final_orbital_rotation = None
        if with_final_orbital_rotation:
            final_orbital_rotation = orbital_rotation_from_parameters(
                params[index:], norb
            )
        return UCJOpSpinless(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=final_orbital_rotation,
        )

    def to_parameters(
        self, *, interaction_pairs: list[tuple[int, int]] | None = None
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
                If specified, `interaction_pairs` should be a list of integer pairs
                representing the orbitals that are allowed to interact. These pairs
                can also be interpreted as indices of diagonal Coulomb matrix entries
                that are allowed to be nonzero.
                Each integer pair must be upper triangular, that is, of the form
                :math:`(i, j)` where :math:`i \leq j`.

        Returns:
            The real-valued parameter vector.

        Raises:
            ValueError: Interaction pairs list contained duplicate interactions.
            ValueError: Interaction pairs list contained lower triangular pairs.
        """
        n_reps, norb, _ = self.diag_coulomb_mats.shape
        n_params = UCJOpSpinless.n_params(
            norb,
            n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=self.final_orbital_rotation is not None,
        )

        triu_indices = cast(
            list[tuple[int, int]],
            list(itertools.combinations_with_replacement(range(norb), 2)),
        )
        if interaction_pairs is None:
            interaction_pairs = triu_indices

        params = np.zeros(n_params)
        index = 0
        for orbital_rotation, diag_coulomb_mat in zip(
            self.orbital_rotations, self.diag_coulomb_mats
        ):
            # Orbital rotation
            n_params = norb**2
            params[index : index + n_params] = orbital_rotation_to_parameters(
                orbital_rotation
            )
            index += n_params
            # Diag Coulomb matrix
            if interaction_pairs:
                n_params = len(interaction_pairs)
                params[index : index + n_params] = diag_coulomb_mat[
                    tuple(zip(*interaction_pairs))
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
        interaction_pairs: list[tuple[int, int]] | None = None,
        tol: float = 1e-8,
    ) -> UCJOpSpinless:
        r"""Initialize the UCJ operator from t2 (and optionally t1) amplitudes.

        Performs a double-factorization of the t2 amplitudes and constructs the
        ansatz repetitions from the terms of the decomposition, up to an optionally
        specified number of ansatz repetitions. Terms are included in decreasing order
        of the absolute value of the corresponding eigenvalue in the factorization.

        Args:
            t2: The t2 amplitudes.
            t1: The t1 amplitudes.
            n_reps: The number of ansatz repetitions.
            interaction_pairs: Optional restrictions on allowed orbital interactions
                for the diagonal Coulomb operators.
                If specified, `interaction_pairs` should be a list of integer pairs
                representing the orbitals that are allowed to interact. These pairs
                can also be interpreted as indices of diagonal Coulomb matrix entries
                that are allowed to be nonzero.
                Each integer pair must be upper triangular, that is, of the form
                :math:`(i, j)` where :math:`i \leq j`.
            tol: Tolerance for error in the double-factorized decomposition of the
                t2 amplitudes.
                The error is defined as the maximum absolute difference between
                an element of the original tensor and the corresponding element of
                the reconstructed tensor.

        Returns:
            The UCJ operator with parameters initialized from the t2 amplitudes.

        Raises:
            ValueError: Interaction pairs list contained duplicate interactions.
            ValueError: Interaction pairs list contained lower triangular pairs.
        """
        _validate_interaction_pairs(interaction_pairs, ordered=False)

        nocc, _, nvrt, _ = t2.shape
        norb = nocc + nvrt

        diag_coulomb_mats, orbital_rotations = linalg.double_factorized_t2(t2, tol=tol)
        diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)[:n_reps]
        orbital_rotations = orbital_rotations.reshape(-1, norb, norb)[:n_reps]

        final_orbital_rotation = None
        if t1 is not None:
            final_orbital_rotation_generator = np.zeros((norb, norb), dtype=complex)
            final_orbital_rotation_generator[:nocc, nocc:] = t1
            final_orbital_rotation_generator[nocc:, :nocc] = -t1.T
            final_orbital_rotation = scipy.linalg.expm(final_orbital_rotation_generator)

        # Zero out diagonal coulomb matrix entries if requested
        if interaction_pairs is not None:
            mask = np.zeros((norb, norb), dtype=bool)
            rows, cols = zip(*interaction_pairs)
            mask[rows, cols] = True
            mask[cols, rows] = True
            diag_coulomb_mats *= mask

        return UCJOpSpinless(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=final_orbital_rotation,
        )

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        if copy:
            vec = vec.copy()
        current_basis = np.eye(norb)
        if isinstance(nelec, int):
            for diag_coulomb_mat, orbital_rotation in zip(
                self.diag_coulomb_mats, self.orbital_rotations
            ):
                vec = gates.apply_orbital_rotation(
                    vec,
                    orbital_rotation.T.conj() @ current_basis,
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
        else:
            zero = np.zeros((norb, norb))
            for diag_coulomb_mat, orbital_rotation in zip(
                self.diag_coulomb_mats, self.orbital_rotations
            ):
                vec = gates.apply_orbital_rotation(
                    vec,
                    orbital_rotation.T.conj() @ current_basis,
                    norb=norb,
                    nelec=nelec,
                    copy=False,
                )
                vec = gates.apply_diag_coulomb_evolution(
                    vec,
                    (diag_coulomb_mat, zero, diag_coulomb_mat),
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
        if isinstance(other, UCJOpSpinless):
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
