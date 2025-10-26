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


def jordan_wigner(op: FermionOperator, norb: int | None = None) -> SparsePauliOp:
    r"""Jordan-Wigner transformation as implemented in `src/fermion_operator.rs`.

    Transform a fermion operator to a qubit operator using the Jordan-Wigner
    transformation. The Jordan-Wigner transformation maps fermionic annihilation
    operators to qubits as follows:

    .. math::

        a_p \mapsto \frac12 (X_p + iY_p)Z_1 \cdots Z_{p-1}

    In the transformed operator, the first ``norb`` qubits represent spin-up (alpha)
    orbitals, and the latter ``norb`` qubits represent spin-down (beta) orbitals. As a
    result of this convention, the qubit index that an orbital is mapped to depends on
    the total number of spatial orbitals. By default, the total number of spatial
    orbitals is automatically determined by the largest-index orbital present in the
    operator, but you can manually specify the number using the `norb` argument.

    Args:
        op: The fermion operator to transform.
        norb: The total number of spatial orbitals. If not specified, it is determined
            by the largest-index orbital present in the operator.

    Returns:
        The qubit operator as a Qiskit SparsePauliOp.

    Raises:
        ValueError: Number of spatial orbitals was negative.
        ValueError: Number of spatial orbitals was fewer than the number detected in the
            operator.
    """
    return op.to_qubit(norb=norb)
