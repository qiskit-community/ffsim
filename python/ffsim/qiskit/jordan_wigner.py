# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Jordan-Wigner transformation."""

from __future__ import annotations

import functools

from qiskit.quantum_info import SparsePauliOp

from ffsim.operators import FermionOperator


def jordan_wigner(op: FermionOperator, n_qubits: int | None = None) -> SparsePauliOp:
    """Jordan-Wigner transformation.

    Transform a fermion operator to a qubit operator using the Jordan-Wigner
    transformation.

    Args:
        op: The fermion operator to transform.
        n_qubits: The number of qubits to include in the output qubit operator. If not
            specified, the minimum number of qubits needed to accommodate the fermion
            operator will be used. Must be non-negative.

    Returns:
        The qubit operator as a Qiskit SparsePauliOp.

    Raises:
        ValueError: Number of qubits was negative.
        ValueError: Number of qubits was not enough to accommodate the fermion operator.
    """
    if n_qubits and n_qubits < 0:
        raise ValueError(f"Number of qubits must be non-negative. Got {n_qubits}.")
    if not op:
        return SparsePauliOp.from_sparse_list([("", [], 0.0)], num_qubits=n_qubits or 0)

    norb = 1 + max(orb for term in op for _, _, orb in term)
    if n_qubits is None:
        n_qubits = 2 * norb
    if n_qubits < 2 * norb:
        raise ValueError(
            "Number of qubits is not enough to accommodate the fermion operator. "
            f"The fermion operator has {norb} spatial orbitals, so at least {2 * norb} "
            f"qubits is needed, but got {n_qubits}."
        )

    qubit_terms = [SparsePauliOp.from_sparse_list([("", [], 0.0)], num_qubits=n_qubits)]
    for term, coeff in op.items():
        qubit_op = SparsePauliOp.from_sparse_list(
            [("", [], coeff)], num_qubits=n_qubits
        )
        for action, spin, orb in term:
            qubit_op @= _qubit_action(action, orb + spin * norb, n_qubits)
        qubit_terms.append(qubit_op)

    return SparsePauliOp.sum(qubit_terms).simplify()


@functools.cache
def _qubit_action(action: bool, qubit: int, n_qubits: int):
    qubits = list(range(qubit + 1))
    return SparsePauliOp.from_sparse_list(
        [
            ("Z" * qubit + "X", qubits, 0.5),
            ("Z" * qubit + "Y", qubits, -0.5j if action else 0.5j),
        ],
        num_qubits=n_qubits,
    )
