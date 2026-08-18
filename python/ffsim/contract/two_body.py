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
from pyscf.fci.direct_uhf import (
    absorb_h1e as absorb_h1e_uhf,
)
from pyscf.fci.direct_uhf import (
    contract_2e as contract_2e_uhf,
)
from scipy.sparse.linalg import LinearOperator

from ffsim import states
from ffsim._cistring import gen_linkstr_index, gen_linkstr_index_trilidx


def two_body_linop(
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    one_body_tensor: np.ndarray | None = None,
    constant: float = 0,
) -> LinearOperator:
    r"""Convert a two-body tensor to a linear operator.

    A two-body tensor has the form

    .. math::

        \frac12 \sum_{\substack{pqrs \\ \sigma \tau}} h_{pqrs}
        a^\dagger_{p\sigma} a^\dagger_{r\tau} a_{s\tau} a_{q\sigma}

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
    if np.iscomplexobj(two_body_tensor) or (
        one_body_tensor is not None and np.iscomplexobj(one_body_tensor)
    ):
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
            two_body_tensor, vec, norb, nelec, link_index=link_index
        )
        if constant:
            result += constant * vec
        return result

    dim_ = states.dim(norb, nelec)
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
            two_body_tensor, vec, norb, nelec, link_index=link_index
        )
        if constant:
            result += constant * vec
        return result

    def rmatvec(vec: np.ndarray):
        result = contract_2e_nosym(
            two_body_tensor.transpose(1, 0, 3, 2).conj(),
            vec,
            norb,
            nelec,
            link_index=link_index,
        )
        if constant:
            result += constant * vec
        return result

    dim_ = states.dim(norb, nelec)
    return LinearOperator(
        shape=(dim_, dim_), matvec=matvec, rmatvec=rmatvec, dtype=complex
    )


def two_body_linop_unrestricted(
    two_body_tensors: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    one_body_tensors: np.ndarray | None = None,
    constant: float = 0,
) -> LinearOperator:
    r"""Convert spin-unrestricted two-body tensors to a linear operator.

    The operator has the form

    .. math::

        \frac12 \sum_{pqrs} h^{\alpha\alpha}_{pqrs}
        a^\dagger_{p\alpha} a^\dagger_{r\alpha} a_{s\alpha} a_{q\alpha}
        + \sum_{pqrs} h^{\alpha\beta}_{pqrs}
        a^\dagger_{p\alpha} a^\dagger_{r\beta} a_{s\beta} a_{q\alpha}
        + \frac12 \sum_{pqrs} h^{\beta\beta}_{pqrs}
        a^\dagger_{p\beta} a^\dagger_{r\beta} a_{s\beta} a_{q\beta}

    where :math:`h^{\alpha\alpha}`, :math:`h^{\alpha\beta}`, and
    :math:`h^{\beta\beta}` are tensors of coefficients in chemist ordering, whose spin
    labels refer to the two *pairs* of indices: in :math:`h^{\alpha\beta}_{pqrs}`, the
    indices :math:`pq` belong to spin alpha and the indices :math:`rs` belong to spin
    beta. The alpha-beta term appears with coefficient 1 rather than :math:`\frac12`
    because the beta-alpha term, whose tensor is
    :math:`h^{\beta\alpha}_{pqrs} = h^{\alpha\beta}_{rspq}`, contributes the other half.
    Consequently, the alpha-beta tensor need not be symmetric under exchanging its two
    index pairs. See :class:`~ffsim.MolecularHamiltonianUnrestricted`.

    Each tensor is assumed to be symmetric within each of its index pairs, that is,
    :math:`h_{pqrs} = h_{qprs} = h_{pqsr}`.

    Args:
        two_body_tensors: The two-body tensors
            :math:`(h^{\alpha\alpha}, h^{\alpha\beta}, h^{\beta\beta})`, as a single
            Numpy array of shape ``(3, norb, norb, norb, norb)``.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        one_body_tensors: Optional one-body tensors
            :math:`(h^{\alpha}, h^{\beta})`, as a single Numpy array of shape
            ``(2, norb, norb)``, to absorb into the two-body operator.
        constant: Optional constant to add to the operator.

    Returns:
        A LinearOperator that implements the action of the two-body tensors.

    Raises:
        NotImplementedError: The tensors have a complex data type. PySCF does not
            provide a spin-unrestricted FCI contraction for complex integrals.
    """
    if np.iscomplexobj(two_body_tensors) or (
        one_body_tensors is not None and np.iscomplexobj(one_body_tensors)
    ):
        raise NotImplementedError(
            "The unrestricted two-body linear operator is not implemented for complex "
            "tensors."
        )
    if one_body_tensors is None:
        one_body_tensors = np.zeros((2, norb, norb))

    n_alpha, n_beta = nelec
    linkstr_index_a = gen_linkstr_index_trilidx(range(norb), n_alpha)
    linkstr_index_b = gen_linkstr_index_trilidx(range(norb), n_beta)
    link_index = (linkstr_index_a, linkstr_index_b)
    absorbed_tensors = absorb_h1e_uhf(
        tuple(np.ascontiguousarray(mat) for mat in one_body_tensors),
        tuple(np.ascontiguousarray(tensor) for tensor in two_body_tensors),
        norb,
        nelec,
        0.5,
    )

    def matvec(vec: np.ndarray):
        # Unlike direct_spin1.contract_2e, direct_uhf.contract_2e does not handle
        # complex vectors, so the real and imaginary parts are contracted separately.
        if np.iscomplexobj(vec):
            result = contract_2e_uhf(
                absorbed_tensors,
                np.ascontiguousarray(vec.real),
                norb,
                nelec,
                link_index=link_index,
            ).astype(complex)
            result.imag = contract_2e_uhf(
                absorbed_tensors,
                np.ascontiguousarray(vec.imag),
                norb,
                nelec,
                link_index=link_index,
            )
        else:
            result = contract_2e_uhf(
                absorbed_tensors, vec, norb, nelec, link_index=link_index
            )
        if constant:
            result += constant * vec
        return result

    dim_ = states.dim(norb, nelec)
    return LinearOperator(
        shape=(dim_, dim_), matvec=matvec, rmatvec=matvec, dtype=complex
    )
