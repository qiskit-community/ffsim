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
    r"""A spin-balanced UCJ operator parameterized by gate rotation angles.

    This operator has the same form as :class:`UCJOpSpinBalanced`,

    .. math::

        \prod_{k = 1}^L \mathcal{U}_k e^{i \mathcal{J}_k} \mathcal{U}_k^\dagger,

    but it is parameterized by the angles of the gates that implement it rather than
    by orbital rotation and diagonal Coulomb matrices.
    Each orbital rotation :math:`\mathcal{U}_k` is represented by a
    :class:`GivensAnsatzOp`, a sequence of Givens rotations followed by a layer of
    single-orbital phase gates. Each diagonal Coulomb evolution
    :math:`e^{i \mathcal{J}_k}` is represented by a
    :class:`NumNumAnsatzOpSpinBalanced`, a sequence of number-number interactions
    whose angles are the entries of the diagonal Coulomb matrices
    :math:`\mathbf{J}^{(\alpha \alpha)}` and :math:`\mathbf{J}^{(\alpha \beta)}`
    at the specified interaction pairs.
    The number of terms :math:`L` is referred to as the
    number of ansatz repetitions and is accessible via the ``n_reps`` attribute.

    To support variational optimization of the orbital basis, an optional final
    orbital rotation can be included in the operator, to be performed at the end.
    """

    norb: int
    """The number of spatial orbitals."""
    num_num_ansatz_ops: list[NumNumAnsatzOpSpinBalanced]
    """The number-number interaction ansatz operators, one per ansatz repetition."""
    givens_ansatz_ops: list[GivensAnsatzOp]
    r"""The Givens rotation ansatz operators, one per ansatz repetition. The operator
    for repetition :math:`k` represents the orbital rotation :math:`\mathcal{U}_k`."""
    final_givens_ansatz_op: GivensAnsatzOp | None = None
    """The optional Givens rotation ansatz operator representing the final orbital
    rotation."""

    def __post_init__(self):
        if len(self.num_num_ansatz_ops) != len(self.givens_ansatz_ops):
            raise ValueError(
                "The number of number-number ansatz operations must equal the number "
                "of Givens ansatz operations. "
                f"Got {len(self.num_num_ansatz_ops)} and {len(self.givens_ansatz_ops)}."
            )

    @property
    def n_reps(self) -> int:
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
        r"""Return the number of parameters of an ansatz with given settings.

        Args:
            norb: The number of spatial orbitals.
            n_reps: The number of ansatz repetitions.
            num_num_interaction_pairs: The orbital pairs to apply the number-number
                interactions to. This should be a pair of lists, for alpha-alpha and
                alpha-beta interactions, in that order.
                Each integer pair must be upper triangular, that is, of the form
                :math:`(i, j)` where :math:`i \leq j`.
            givens_interaction_pairs: The orbital pairs to apply the Givens rotations
                to in each ansatz repetition.
            with_final_givens_ansatz_op: Whether to include a final Givens rotation
                ansatz operator. The final operator uses a full brickwork pattern of
                Givens rotations, so it contributes ``norb**2`` parameters.

        Returns:
            The number of parameters of the ansatz.
        """
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
        r"""Initialize the UCJ operator from a real-valued parameter vector.

        For each ansatz repetition, the parameter vector contains the parameters of
        the Givens rotation ansatz operator followed by the parameters of the
        number-number interaction ansatz operator. These are followed by the
        parameters of the final Givens rotation ansatz operator, if present.
        See :meth:`GivensAnsatzOp.to_parameters` and
        :meth:`NumNumAnsatzOpSpinBalanced.to_parameters` for the parameter orderings
        of the individual operators.

        Args:
            params: The real-valued parameter vector.
            norb: The number of spatial orbitals.
            n_reps: The number of ansatz repetitions.
            num_num_interaction_pairs: The orbital pairs to apply the number-number
                interactions to. This should be a pair of lists, for alpha-alpha and
                alpha-beta interactions, in that order.
                Each integer pair must be upper triangular, that is, of the form
                :math:`(i, j)` where :math:`i \leq j`.
            givens_interaction_pairs: The orbital pairs to apply the Givens rotations
                to in each ansatz repetition. Operators constructed with
                :meth:`from_ucj_op` or :meth:`from_t_amplitudes` use a full brickwork
                pattern of nearest-neighbor pairs with ``norb`` layers.
            with_final_givens_ansatz_op: Whether to include a final Givens rotation
                ansatz operator. The final operator uses a full brickwork pattern of
                Givens rotations.

        Returns:
            The UCJ operator constructed from the given parameters.

        Raises:
            ValueError: The number of parameters passed did not match the number
                expected based on the function inputs.
        """
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
        r"""Convert the UCJ operator to a real-valued parameter vector.

        See :meth:`from_parameters` for the ordering of the parameters.

        Returns:
            The real-valued parameter vector.
        """
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
    def from_cisd_vec(
        cisd_vec: np.ndarray,
        *,
        norb: int,
        nocc: int,
        c0_threshold: float = 1e-12,
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
        r"""Initialize the UCJ operator from a CISD vector.

        Constructs a :class:`UCJOpSpinBalanced` using
        :meth:`UCJOpSpinBalanced.from_cisd_vec` and converts it using
        :meth:`from_ucj_op`. See :meth:`UCJOpSpinBalanced.from_cisd_vec` for how the
        CISD amplitudes are converted to CC-style amplitudes.

        Args:
            cisd_vec: The CISD vector. This is a one-dimensional array containing the
                reference coefficient :math:`c_0` in the first entry, followed by the
                singles and then doubles coefficients.
            norb: The number of spatial orbitals.
            nocc: The number of occupied orbitals.
            c0_threshold: Absolute value threshold for the reference coefficient. An
                error is raised if the absolute value of :math:`c_0` is smaller than
                this threshold.
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
                converted t2 amplitudes.
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
            validate: Whether to validate the symmetry of the converted t2
                amplitudes.
            rtol: Relative numerical tolerance for validation checks.
            atol: Absolute numerical tolerance for validation checks.
            optimize_kwargs: Keyword arguments to pass to
                :func:`scipy.optimize.minimize`. The default method is
                ``"L-BFGS-B"``.
                These arguments are ignored if ``optimize`` is set to ``False``.

        Returns:
            The UCJ operator with parameters initialized from the CISD amplitudes.

        Raises:
            ValueError: The CISD reference coefficient is smaller than the
                specified threshold c0_threshold.
            ValueError: Interaction pairs list contained duplicate interactions.
            ValueError: Interaction pairs list contained out-of-range orbital indices.
            ValueError: Interaction pairs list contained lower triangular pairs.
            ValueError: The converted t2 amplitudes did not have the required
                symmetry.
        """
        ucj_op = UCJOpSpinBalanced.from_cisd_vec(
            cisd_vec,
            norb=norb,
            nocc=nocc,
            c0_threshold=c0_threshold,
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
    r"""Yield the orbital pairs of a brickwork pattern of nearest-neighbor gates.

    Even-numbered layers act on pairs :math:`(0, 1), (2, 3), \ldots` and odd-numbered
    layers act on pairs :math:`(1, 2), (3, 4), \ldots`.
    """
    for i in range(n_layers):
        for j in range(i % 2, norb - 1, 2):
            yield (j, j + 1)
