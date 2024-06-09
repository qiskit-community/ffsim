# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Spin-balanced number-number interaction ansatz."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ffsim import gates


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
                    "You must provide only upper triangular interaction pairs. "
                    f"Got {(i, j)}, which is a lower triangular pair."
                )


@dataclass(frozen=True)
class NumNumAnsatzOpSpinBalanced:
    """A number-number interaction ansatz operator.

    The number-number interaction ansatz consists of a sequence of
    `number-number interactions`_.

    Attributes:
        norb (int): The number of spatial orbitals.
        interaction_pairs (list[tuple[int, int]]): The orbital pairs to apply the
            number-number interactions to.
        thetas (np.ndarray): The angles for the number-number interactions.

    .. _number-number interactions: ffsim.html#ffsim.apply_num_num_interaction
    """

    norb: int
    interaction_pairs: tuple[list[tuple[int, int]], list[tuple[int, int]]]
    thetas: tuple[np.ndarray, np.ndarray]

    def __post_init__(self):
        for pairs, angles in zip(self.interaction_pairs, self.thetas):
            _validate_interaction_pairs(pairs, ordered=False)
            if not len(pairs) == len(angles):
                raise ValueError(
                    "The number of interaction pairs must be equal to the number of "
                    "rotation angles."
                    f"Got {len(pairs)}, and {len(angles)}."
                )

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        """Apply the operator to a vector."""
        if isinstance(nelec, int):
            return NotImplemented
        mats_aa, mats_ab = self.to_diag_coulomb_mats()
        return gates.apply_diag_coulomb_evolution(
            vec,
            (mats_aa, mats_ab, mats_aa),
            time=-1.0,
            norb=norb,
            nelec=nelec,
            copy=copy,
        )

    @staticmethod
    def n_params(interaction_pairs: list[tuple[int, int]]) -> int:
        """Return the number of parameters of an ansatz with given settings.

        Args:
            interaction_pairs: The orbital pairs to apply the number-number interactions
                to.
        """
        return sum(len(pairs) for pairs in interaction_pairs)

    def to_parameters(self) -> np.ndarray:
        """Convert the operator to a real-valued parameter vector."""
        return np.concatenate(self.thetas)

    @staticmethod
    def from_parameters(
        params: np.ndarray,
        norb: int,
        interaction_pairs: tuple[list[tuple[int, int]], list[tuple[int, int]]],
    ) -> NumNumAnsatzOpSpinBalanced:
        """Initialize the operator from a real-valued parameter vector.

        Args:
            params: The real-valued parameter vector.
            norb: The number of spatial orbitals.
            interaction_pairs: The orbital pairs to apply the number-number interactions
                to.
        """
        pairs_aa, pairs_ab = interaction_pairs
        n_params = len(pairs_aa) + len(pairs_ab)
        if len(params) != n_params:
            raise ValueError(
                "The number of parameters passed did not match the number expected "
                "based on the function inputs. "
                f"Expected {n_params} but got {len(params)}."
            )
        thetas_aa = params[: len(pairs_aa)]
        thetas_ab = params[len(pairs_aa) :]
        return NumNumAnsatzOpSpinBalanced(
            norb=norb,
            interaction_pairs=interaction_pairs,
            thetas=(thetas_aa, thetas_ab),
        )

    @staticmethod
    def from_diag_coulomb_mats(
        diag_coulomb_mats: tuple[np.ndarray, np.ndarray] | np.ndarray,
    ) -> NumNumAnsatzOpSpinBalanced:
        """Initialize the operator from a diagonal Coulomb matrix.

        Args:
            diag_coulomb_mats: The diagonal Coulomb matrices. Should be a pair
                of matrices, with the first matrix representing same-spin interactions
                and the second matrix representing different-spin interactions.
        """
        mat_aa, mat_ab = diag_coulomb_mats
        norb, _ = mat_aa.shape
        pairs_aa: list[tuple[int, int]] = []
        pairs_ab: list[tuple[int, int]] = []
        thetas_aa: list[float] = []
        thetas_ab: list[float] = []
        for mat, pairs, thetas in [
            (mat_aa, pairs_aa, thetas_aa),
            (mat_ab, pairs_ab, thetas_ab),
        ]:
            for pair in itertools.combinations_with_replacement(range(norb), 2):
                val = mat[pair]
                if val:
                    pairs.append(pair)
                    thetas.append(val)

        return NumNumAnsatzOpSpinBalanced(
            norb=norb,
            interaction_pairs=(pairs_aa, pairs_ab),
            thetas=(np.array(thetas_aa), np.array(thetas_ab)),
        )

    def to_diag_coulomb_mats(self) -> np.ndarray:
        """Convert the operator to diagonal Coulomb matrices.

        Returns:
            A Numpy array of shape (2, norb, norb) holding two matrices. The first
            matrix holds the same-spin interactions and the second matrix holds the
            different-spin interactions.
        """
        diag_coulomb_mats = np.zeros((2, self.norb, self.norb))
        for mat, pairs, thetas in zip(
            diag_coulomb_mats, self.interaction_pairs, self.thetas
        ):
            for (i, j), theta in zip(pairs, thetas):
                mat[i, j] = theta
                mat[j, i] = theta
        return diag_coulomb_mats

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        if isinstance(other, NumNumAnsatzOpSpinBalanced):
            if self.norb != other.norb:
                return False
            if self.interaction_pairs != other.interaction_pairs:
                return False
            if not np.allclose(self.thetas[0], other.thetas[0], rtol=rtol, atol=atol):
                return False
            if not np.allclose(self.thetas[1], other.thetas[1], rtol=rtol, atol=atol):
                return False
            return True
        return NotImplemented
