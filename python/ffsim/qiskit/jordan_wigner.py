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
from typing import Iterable, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed

from qiskit.quantum_info import SparsePauliOp

from ffsim.operators import FermionOperator



def jordan_wigner(
    op,
    norb: int | None = None,
    parallel: bool = False,          
    max_workers: int | None = None,  
    chunk: int = 256,                
):
    """Map a fermion operator to a qubit operator using the Jordan–Wigner transform.

    Args:
        op: The fermion operator to transform.
        norb: The total number of spatial orbitals. If not specified, it is determined
            by the largest-index orbital present in the operator.
        parallell: Allow for parallel execution. Default is False.
        max_workers: The maximum number of worker processes to use if parallel is True.
        chunk: The number of terms to include in each parallel chunk if parallel is True.

    Returns:
        The qubit operator as a Qiskit SparsePauliOp.

    Raises:
        ValueError: Number of spatial orbitals was negative.
        ValueError: Number of spatial orbitals was fewer than the number detected in the
            operator.
    """
    if norb is not None and norb < 0:
        raise ValueError(f"Number of spatial orbitals must be non-negative. Got {norb}.")

    if not op:
        return SparsePauliOp.from_sparse_list([("", [], 0.0)], num_qubits=2 * (norb or 0))

    norb_in_op = 1 + max(orb for term in op for _, _, orb in term)
    if norb is None:
        norb = norb_in_op
    if norb < norb_in_op:
        raise ValueError(
            "Number of spatial orbitals specified is fewer than the number detected in the operator."
        )

    num_qubits = 2 * norb

    # -- Parallel version --
    if parallel:
        items = list(op.items())
        if len(items) <= max(chunk, 1):
            qubit_terms = [SparsePauliOp.from_sparse_list([("", [], 0.0)], num_qubits=num_qubits)]
            for term, coeff in items:
                qubit_terms.append(_term_to_qubit_op((term, coeff, norb)))
            return SparsePauliOp.sum(qubit_terms).simplify()

        partial_sums = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    _sum_batch,
                    [(term, coeff, norb) for term, coeff in batch],
                    num_qubits,
                )
                for batch in _chunks(items, max(chunk, 1))
            ]
            for fut in as_completed(futures):
                partial_sums.append(fut.result())

        return SparsePauliOp.sum(partial_sums).simplify()

    qubit_terms = [SparsePauliOp.from_sparse_list([("", [], 0.0)], num_qubits=num_qubits)]
    for term, coeff in op.items():
        qubit_op = SparsePauliOp.from_sparse_list([("", [], coeff)], num_qubits=num_qubits)
        for action, spin, orb in term:
            qubit_op @= _qubit_action(action, orb + spin * norb, norb)
        qubit_terms.append(qubit_op)

    return SparsePauliOp.sum(qubit_terms).simplify()


@functools.cache
def _qubit_action(action: bool, qubit: int, norb: int):
    qubits = list(range(qubit + 1))
    return SparsePauliOp.from_sparse_list(
        [
            ("Z" * qubit + "X", qubits, 0.5),
            ("Z" * qubit + "Y", qubits, -0.5j if action else 0.5j),
        ],
        num_qubits=2 * norb,
    )

def _term_to_qubit_op(term_coeff_norb: Tuple[tuple, complex, int]) -> SparsePauliOp:
    term, coeff, norb = term_coeff_norb
    qubit_op = SparsePauliOp.from_sparse_list([("", [], coeff)], num_qubits=2 * norb)
    # term is an ordered tuple of (action: bool, spin: int, orb: int)
    for action, spin, orb in term:
        qubit_op @= _qubit_action(action, orb + spin * norb, norb)
    return qubit_op


def _sum_batch(batch: List[Tuple[tuple, complex, int]], num_qubits: int) -> SparsePauliOp:
    # Local reduction of a batch of terms to a SparsePauliOp
    qubit_terms = [SparsePauliOp.from_sparse_list([("", [], 0.0)], num_qubits=num_qubits)]
    for term, coeff, norb in batch:
        qubit_terms.append(_term_to_qubit_op((term, coeff, norb)))
    return SparsePauliOp.sum(qubit_terms)


def _chunks(it: Iterable, size: int):
    it = iter(it)
    while True:
        batch = []
        try:
            for _ in range(size):
                batch.append(next(it))
        except StopIteration:
            if batch:
                yield batch
            break
        yield batch
