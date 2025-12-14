# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Contract two-body operator."""

from __future__ import annotations

import numpy as np
from pyscf.fci.direct_nosym import (
    absorb_h1e as absorb_h1e_nosym,
)
from pyscf.fci.direct_nosym import (
    contract_2e as contract_2e_nosym,
)
from pyscf.fci.direct_spin1 import (
    absorb_h1e as absorb_h1e_spin1,
)
from pyscf.fci.direct_spin1 import (
    contract_2e as contract_2e_spin1,
)
from scipy.sparse.linalg import LinearOperator

from ffsim import dimensions
from ffsim.cistring import gen_linkstr_index, gen_linkstr_index_trilidx


def two_body_linop(
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    one_body_tensor: np.ndarray | None = None,
    constant: float = 0,
) -> LinearOperator:
    if np.iscomplexobj(two_body_tensor) or np.iscomplexobj(one_body_tensor):
        return _two_body_linop_complex(
            two_body_tensor,
            norb=norb,
            nelec=nelec,
            one_body_tensor=one_body_tensor,
            constant=constant,
        )
    return _two_body_linop_real(
        two_body_tensor,
        norb=norb,
        nelec=nelec,
        one_body_tensor=one_body_tensor,
        constant=constant,
    )


def _two_body_linop_real(
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    one_body_tensor: np.ndarray | None = None,
    constant: float = 0,
) -> LinearOperator:
    r"""Convert a two-body tensor to a linear operator.

    A two-body tensor has the form

    .. math::

        \sum_{\sigma \tau, pqrs} h_{pqrs}
        a^\dagger_{\sigma, p} a^\dagger_{\tau, r} a_{\tau, s} a_{\sigma, q}

    where :math:`h_{pqrs}` is a tensor of complex coefficients.

    Args:
        two_body_tensor: The two-body tensor.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        one_body_tensor: Optional one-body tensor to absorb into the two-body operator.
            See :func:`~.one_body_linop`.
        constant: Optional constant to add to the operator.

    Returns:
        A LinearOperator that implements the action of the two-body tensor.
    """
    if one_body_tensor is None:
        one_body_tensor = np.zeros((norb, norb))

    n_alpha, n_beta = nelec
    linkstr_index_a = gen_linkstr_index_trilidx(range(norb), n_alpha)
    linkstr_index_b = gen_linkstr_index_trilidx(range(norb), n_beta)
    link_index = (linkstr_index_a, linkstr_index_b)
    two_body_tensor = absorb_h1e_spin1(
        one_body_tensor, two_body_tensor, norb, nelec, 0.5
    )

    def matvec(vec: np.ndarray):
        result = contract_2e_spin1(
            two_body_tensor,
            vec.real,
            norb,
            nelec,
            link_index=link_index,
        ).astype(complex)
        result += 1j * contract_2e_spin1(
            two_body_tensor,
            vec.imag,
            norb,
            nelec,
            link_index=link_index,
        )
        if constant:
            result += constant * vec
        return result

    dim_ = dimensions.dim(norb, nelec)
    return LinearOperator(
        shape=(dim_, dim_), matvec=matvec, rmatvec=matvec, dtype=complex
    )


def _two_body_linop_complex(
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    one_body_tensor: np.ndarray | None = None,
    constant: float = 0,
) -> LinearOperator:
    r"""Convert a two-body tensor to a linear operator.

    A two-body tensor has the form

    .. math::

        \sum_{\sigma \tau, pqrs} h_{pqrs}
        a^\dagger_{\sigma, p} a^\dagger_{\tau, r} a_{\tau, s} a_{\sigma, q}

    where :math:`h_{pqrs}` is a tensor of complex coefficients.

    Args:
        two_body_tensor: The two-body tensor.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        one_body_tensor: Optional one-body tensor to absorb into the two-body operator.
            See :func:`~.one_body_linop`.
        constant: Optional constant to add to the operator.

    Returns:
        A LinearOperator that implements the action of the two-body tensor.
    """
    if one_body_tensor is None:
        one_body_tensor = np.zeros((norb, norb))

    n_alpha, n_beta = nelec
    linkstr_index_a = gen_linkstr_index(range(norb), n_alpha)
    linkstr_index_b = gen_linkstr_index(range(norb), n_beta)
    link_index = (linkstr_index_a, linkstr_index_b)
    two_body_tensor = absorb_h1e_nosym(
        one_body_tensor, two_body_tensor, norb, nelec, 0.5
    )

    def matvec(vec: np.ndarray):
        result = contract_2e_nosym(
            two_body_tensor,
            vec,
            norb,
            nelec,
            link_index=link_index,
        )
        if constant:
            result += constant * vec
        return result

    def rmatvec(vec: np.ndarray):
        result = contract_2e_nosym(
            # TODO come up with a way to test this transpose
            two_body_tensor.transpose(1, 0, 3, 2).conj(),
            vec,
            norb,
            nelec,
            link_index=link_index,
        )
        if constant:
            result += constant * vec
        return result

    dim_ = dimensions.dim(norb, nelec)
    return LinearOperator(
        shape=(dim_, dim_), matvec=matvec, rmatvec=rmatvec, dtype=complex
    )
