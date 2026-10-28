# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Spin-balanced (local) UCJ ansatz parameterized by gate rotation angles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ffsim import gates, protocols
from ffsim.linalg.util import rows_and_cols, upper_triangular_indices
from ffsim.variational.givens import GivensAnsatzOp
from ffsim.variational.num_num import NumNumAnsatzOpSpinBalanced
from ffsim.variational.ucj_spin_balanced import UCJOpSpinBalanced
from ffsim.variational.util import validate_interaction_pairs


@dataclass(frozen=True)
class UCJAnglesOpSpinBalanced(
    protocols.SupportsApplyUnitary, protocols.SupportsApproximateEquality
):
    r"""A spin-balanced UCJ operator parameterized by gate rotation angles."""

    norb: int
    num_num_ansatz_ops: list[NumNumAnsatzOpSpinBalanced]
    givens_ansatz_ops: list[GivensAnsatzOp]
    final_givens_ansatz_op: GivensAnsatzOp | None = None

    def __post_init__(self):
        if len(self.num_num_ansatz_ops) != len(self.givens_ansatz_ops):
            raise ValueError(
                "The number of number-number ansatz operations must equal the number "
                "of Givens ansatz operations. "
                f"Got {len(self.num_num_ansatz_ops)} and {len(self.givens_ansatz_ops)}."
            )

    @property
    def n_reps(self):
        """The number of ansatz repetitions."""
        return len(self.num_num_ansatz_ops)

    @staticmethod
    def n_params(
        norb: int,
        n_reps: int,
        num_num_interaction_pairs: tuple[list[tuple[int, int]], list[tuple[int, int]]],
        givens_interaction_pairs: list[tuple[int, int]],
        with_final_givens_ansatz_op: bool = False,
    ) -> int:
        """Return the number of parameters of an ansatz with given settings."""
        pairs_aa, pairs_ab = num_num_interaction_pairs
        n_params_num_num = len(pairs_aa) + len(pairs_ab)
        n_params_givens = 2 * len(givens_interaction_pairs) + norb
        return (
            n_reps * (n_params_num_num + n_params_givens)
            + with_final_givens_ansatz_op * norb**2
        )

    @staticmethod
    def from_parameters(
        params: np.ndarray,
        *,
        norb: int,
        n_reps: int,
        num_num_interaction_pairs: tuple[list[tuple[int, int]], list[tuple[int, int]]],
        givens_interaction_pairs: list[tuple[int, int]],
        with_final_givens_ansatz_op: bool = False,
    ) -> UCJAnglesOpSpinBalanced:
        r"""Initialize the UCJ operator from a real-valued parameter vector."""
        n_params = UCJAnglesOpSpinBalanced.n_params(
            norb,
            n_reps,
            num_num_interaction_pairs=num_num_interaction_pairs,
            givens_interaction_pairs=givens_interaction_pairs,
            with_final_givens_ansatz_op=with_final_givens_ansatz_op,
        )
        if len(params) != n_params:
            raise ValueError(
                "The number of parameters passed did not match the number expected "
                "based on the function inputs. "
                f"Expected {n_params} but got {len(params)}."
            )
        n_params_num_num = sum(len(pairs) for pairs in num_num_interaction_pairs)
        n_params_givens = 2 * len(givens_interaction_pairs) + norb
        num_num_ansatz_ops = []
        givens_ansatz_ops = []
        index = 0
        for _ in range(n_reps):
            # Givens
            n_params = n_params_givens
            givens_ansatz_ops.append(
                GivensAnsatzOp.from_parameters(
                    params[index : index + n_params],
                    norb=norb,
                    interaction_pairs=givens_interaction_pairs,
                )
            )
            index += n_params
            # Number-number
            n_params = n_params_num_num
            num_num_ansatz_ops.append(
                NumNumAnsatzOpSpinBalanced.from_parameters(
                    params[index : index + n_params],
                    norb=norb,
                    interaction_pairs=num_num_interaction_pairs,
                )
            )
            index += n_params
        final_givens_ansatz_op = None
        if with_final_givens_ansatz_op:
            final_givens_ansatz_op = GivensAnsatzOp.from_parameters(
                params[index:],
                norb=norb,
                interaction_pairs=list(brickwork(norb, norb)),
            )
        return UCJAnglesOpSpinBalanced(
            norb,
            num_num_ansatz_ops=num_num_ansatz_ops,
            givens_ansatz_ops=givens_ansatz_ops,
            final_givens_ansatz_op=final_givens_ansatz_op,
        )

    def to_parameters(self) -> np.ndarray:
        r"""Convert the UCJ operator to a real-valued parameter vector."""
        param_arrays = []
        for givens_ansatz, num_num_ansatz in zip(
            self.givens_ansatz_ops, self.num_num_ansatz_ops
        ):
            param_arrays.append(givens_ansatz.to_parameters())
            param_arrays.append(num_num_ansatz.to_parameters())
        if self.final_givens_ansatz_op is not None:
            param_arrays.append(self.final_givens_ansatz_op.to_parameters())
        return np.concatenate(param_arrays)

    @staticmethod
    def from_t_amplitudes(
        t2: np.ndarray,
        *,
        t1: np.ndarray | None = None,
        n_reps: int | None = None,
        interaction_pairs: tuple[
            list[tuple[int, int]] | None, list[tuple[int, int]] | None
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
    ) -> UCJAnglesOpSpinBalanced:
        r"""Initialize the UCJ operator from t2 (and optionally t1) amplitudes.

        Constructs a :class:`UCJOpSpinBalanced` using
        :meth:`UCJOpSpinBalanced.from_t_amplitudes` and converts it using
        :meth:`from_ucj_op`.

        Args:
            t2: The t2 amplitudes. They must satisfy
                ``t2[i, j, a, b] == t2[j, i, b, a]``.
            t1: The t1 amplitudes.
            n_reps: The number of ansatz repetitions. If not specified, then it is set
                to the number of terms resulting from the double factorization of the
                t2 amplitudes. If the specified number of repetitions is larger than the
                number of terms resulting from the double factorization, then the ansatz
                is padded with additional identity operators up to the specified number
                of repetitions.
            interaction_pairs: Optional restrictions on allowed orbital interactions
                for the number-number interactions.
                If specified, ``interaction_pairs`` should be a pair of lists,
                for alpha-alpha and alpha-beta interactions, in that order.
                Either list can be substituted with ``None`` to indicate no restrictions
                on interactions.
                Each list should contain pairs of integers representing the orbitals
                that are allowed to interact.
                Each integer pair must be upper triangular, that is, of the form
                :math:`(i, j)` where :math:`i \leq j`.
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
            ValueError: Interaction pairs list contained lower triangular pairs.
            ValueError: The t2 amplitudes did not have the required symmetry.
        """
        ucj_op = UCJOpSpinBalanced.from_t_amplitudes(
            t2,
            t1=t1,
            n_reps=n_reps,
            interaction_pairs=interaction_pairs,
            tol=tol,
            optimize=optimize,
            regularization=regularization,
            multi_stage_start=multi_stage_start,
            multi_stage_step=multi_stage_step,
            validate=validate,
            rtol=rtol,
            atol=atol,
            **optimize_kwargs,
        )
        return UCJAnglesOpSpinBalanced.from_ucj_op(
            ucj_op, interaction_pairs=interaction_pairs
        )

    @staticmethod
    def from_ucj_op(
        ucj_op: UCJOpSpinBalanced,
        *,
        interaction_pairs: tuple[
            list[tuple[int, int]] | None, list[tuple[int, int]] | None
        ]
        | None = None,
    ) -> UCJAnglesOpSpinBalanced:
        r"""Initialize the angles-based UCJ operator from a matrix-based UCJ operator.

        Note:
            If ``interaction_pairs`` is specified, the returned operator will
            incorporate only the diagonal Coulomb matrix entries corresponding to the
            specified interactions, so it will not be equivalent to the original
            operator unless the other entries are zero.

        Args:
            ucj_op: The UCJ operator.
            interaction_pairs: Optional restrictions on allowed orbital interactions
                for the number-number interactions.
                If specified, ``interaction_pairs`` should be a pair of lists,
                for alpha-alpha and alpha-beta interactions, in that order.
                Either list can be substituted with ``None`` to indicate no restrictions
                on interactions, in which case all upper triangular pairs are used.
                Each list should contain pairs of integers representing the orbitals
                that are allowed to interact. The order of the pairs determines the
                order of the corresponding parameters.
                Each integer pair must be upper triangular, that is, of the form
                :math:`(i, j)` where :math:`i \leq j`.

        Returns:
            The angles-based UCJ operator.

        Raises:
            ValueError: Interaction pairs list contained duplicate interactions.
            ValueError: Interaction pairs list contained out-of-range orbital indices.
            ValueError: Interaction pairs list contained lower triangular pairs.
        """
        norb = ucj_op.norb
        if interaction_pairs is None:
            interaction_pairs = (None, None)
        pairs_aa, pairs_ab = interaction_pairs
        validate_interaction_pairs(pairs_aa, ordered=False, norb=norb)
        validate_interaction_pairs(pairs_ab, ordered=False, norb=norb)
        # Fix the pairs up front so that the parameter layout does not depend on
        # which diagonal Coulomb matrix entries happen to be zero
        triu_indices = upper_triangular_indices(norb)
        if pairs_aa is None:
            pairs_aa = triu_indices
        if pairs_ab is None:
            pairs_ab = triu_indices
        rows_aa, cols_aa = rows_and_cols(pairs_aa)
        rows_ab, cols_ab = rows_and_cols(pairs_ab)
        num_num_ansatz_ops = [
            NumNumAnsatzOpSpinBalanced(
                norb=norb,
                interaction_pairs=(pairs_aa, pairs_ab),
                thetas=(mat_aa[rows_aa, cols_aa], mat_ab[rows_ab, cols_ab]),
            )
            for mat_aa, mat_ab in ucj_op.diag_coulomb_mats
        ]
        givens_ansatz_ops = [
            GivensAnsatzOp.from_orbital_rotation(orbital_rotation)
            for orbital_rotation in ucj_op.orbital_rotations
        ]
        final_givens_ansatz_op = None
        if ucj_op.final_orbital_rotation is not None:
            final_givens_ansatz_op = GivensAnsatzOp.from_orbital_rotation(
                ucj_op.final_orbital_rotation
            )
        return UCJAnglesOpSpinBalanced(
            norb=ucj_op.norb,
            num_num_ansatz_ops=num_num_ansatz_ops,
            givens_ansatz_ops=givens_ansatz_ops,
            final_givens_ansatz_op=final_givens_ansatz_op,
        )

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        if isinstance(nelec, int):
            return NotImplemented
        if copy:
            vec = vec.copy()
        current_basis = np.eye(norb)
        for num_num_ansatz_op, givens_ansatz_op in zip(
            self.num_num_ansatz_ops, self.givens_ansatz_ops
        ):
            orbital_rotation = givens_ansatz_op.to_orbital_rotation()
            vec = gates.apply_orbital_rotation(
                vec,
                orbital_rotation.T.conj() @ current_basis,
                norb=norb,
                nelec=nelec,
                copy=False,
            )
            vec = protocols.apply_unitary(
                vec, num_num_ansatz_op, norb=norb, nelec=nelec, copy=False
            )
            current_basis = orbital_rotation
        if self.final_givens_ansatz_op is None:
            vec = gates.apply_orbital_rotation(
                vec, current_basis, norb=norb, nelec=nelec, copy=False
            )
        else:
            final_orbital_rotation = self.final_givens_ansatz_op.to_orbital_rotation()
            vec = gates.apply_orbital_rotation(
                vec,
                final_orbital_rotation @ current_basis,
                norb=norb,
                nelec=nelec,
                copy=False,
            )
        return vec

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        if isinstance(other, UCJAnglesOpSpinBalanced):
            if self.norb != other.norb:
                return False
            if len(self.num_num_ansatz_ops) != len(other.num_num_ansatz_ops):
                return False
            if len(self.givens_ansatz_ops) != len(other.givens_ansatz_ops):
                return False
            if not protocols.approx_eq(
                self.final_givens_ansatz_op,
                other.final_givens_ansatz_op,
                rtol=rtol,
                atol=atol,
            ):
                return False
            if not all(
                protocols.approx_eq(op1, op2, rtol=rtol, atol=atol)
                for op1, op2 in zip(self.num_num_ansatz_ops, other.num_num_ansatz_ops)
            ):
                return False
            if not all(
                protocols.approx_eq(op1, op2, rtol=rtol, atol=atol)
                for op1, op2 in zip(self.givens_ansatz_ops, other.givens_ansatz_ops)
            ):
                return False
            return True
        return NotImplemented


def brickwork(norb: int, n_layers: int):
    for i in range(n_layers):
        for j in range(i % 2, norb - 1, 2):
            yield (j, j + 1)
