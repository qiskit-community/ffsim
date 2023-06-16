# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import itertools
import string
from collections.abc import Sequence

import numpy as np


def expectation_product(
    one_body_tensors: Sequence[np.ndarray],
    one_rdm: np.ndarray,
) -> complex:
    r"""Expectation of product of one-body operators w.r.t. a Slater determinant.

    A one-body operator :math:`O` has the form

    .. math::

        O = \sum_{pq} M_{pq} a_p^\dagger a_q.

    This function takes a list of the matrices :math:`M` as its first argument.
    Let :math:`(O_1, O_2, \dots O_k)` be the list of one-body operators,
    and :math:`\lvert \psi \rangle` be the Slater determinant.
    Then this function returns the quantity

    .. math::

        \langle \psi \rvert O_1 O_2 \dots O_k \lvert \psi \rangle.

    Args:
        one_body_tensors: The matrices for the one-body operators.
        one_rdm: The one-body reduced density matrix of the Slater determinant.

    Returns:
        The expectation value.
    """
    n_tensors = len(one_body_tensors)
    if not n_tensors:
        return 1.0

    norb, _ = one_rdm.shape
    anti_one_rdm = np.eye(norb) - one_rdm

    alphabet = string.ascii_uppercase + string.ascii_lowercase
    indices = alphabet[: 2 * n_tensors]
    creation_indices = indices[0::2]
    annihilation_indices = indices[1::2]

    result = 0.0
    for perm in itertools.permutations(annihilation_indices):
        tensors = []
        pairings = list(
            itertools.chain.from_iterable(zip(creation_indices, annihilation_indices))
        )
        subscripts = [
            f"{c_i}{a_i}" for (c_i, a_i) in zip(creation_indices, annihilation_indices)
        ]
        sign = 1
        for c_i, a_i in zip(creation_indices, perm):
            sign *= (-1) ** (pairings.index(c_i) - pairings.index(a_i) + 1)
            pairings.remove(c_i)
            pairings.remove(a_i)
            subscripts.append(f"{c_i}{a_i}")
            tensors.append(
                one_rdm if indices.index(c_i) < indices.index(a_i) else anti_one_rdm
            )
        result += sign * np.einsum(
            f'{",".join(subscripts)}->',
            *one_body_tensors,
            *tensors[:n_tensors],
            optimize=True,
        )
    return result


def expectation_power(
    one_body_tensor: np.ndarray,
    one_rdm: np.ndarray,
    power: int = 1,
) -> complex:
    r"""Expectation of power of one-body operator w.r.t. a Slater determinant.

    A one-body operator :math:`O` has the form

    .. math::

        O = \sum_{pq} M_{pq} a_p^\dagger a_q.

    This function takes the matrix :math:`M` as its first argument.
    Let :math:`\lvert \psi \rangle` be the Slater determinant.
    Then this function returns the quantity

    .. math::

        \langle \psi \rvert O^k \lvert \psi \rangle.

    Args:
        one_body_tensor: The one-body operator.
        one_rdm: The one-body reduced density matrix of the Slater determinant.
        power: The power.

    Returns:
        The expectation value.
    """
    return expectation_product([one_body_tensor] * power, one_rdm)
