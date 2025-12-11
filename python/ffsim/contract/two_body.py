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

import ctypes

import numpy
import numpy as np
from pyscf import ao2mo
from pyscf.fci.cistring import libfci
from pyscf.fci.direct_spin1 import FCIvector, _unpack, absorb_h1e
from scipy.sparse.linalg import LinearOperator

from ffsim import dimensions
from ffsim.cistring import gen_linkstr_index_trilidx


def contract_2e(eri, fcivec, norb, nelec, link_index=None):
    r"""Contract the 4-index tensor eri[pqrs] with a FCI vector

    .. math::

        |output\rangle = E_{pq} E_{rs} eri_{pq,rs} |CI\rangle \\

        E_{pq}E_{rs} = E_{pr,qs} + \delta_{qr} E_{ps} \\

        E_{pq} = p^+ q + \bar{p}^+ \bar{q}

        E_{pr,qs} = p^+ r^+ s q + \bar{p}^+ r^+ s \bar{q} + ...

    :math:`p,q,...` means spin-up orbitals and :math:`\bar{p}, \bar{q}` means
    spin-down orbitals.

    Note the input argument eri is NOT the 2e hamiltonian tensor. 2e hamiltonian is

    .. math::

        h2e &= (pq|rs) E_{pr,qs} \\
            &= (pq|rs) (E_{pq}E_{rs} - \delta_{qr} E_{ps}) \\
            &= eri_{pq,rs} E_{pq}E_{rs} \\

    So the relation between eri and hamiltonian (the 2e-integral tensor) is

    .. math::

        eri_{pq,rs} = (pq|rs) - (1/Nelec) \sum_q (pq|qs)

    to restore the symmetry between pq and rs,

    .. math::

        eri_{pq,rs} = (pq|rs) - (.5/Nelec) [\sum_q (pq|qs) + \sum_p (pq|rp)]

    See also :func:`direct_spin1.absorb_h1e`
    """
    fcivec = np.asarray(fcivec, order="C")
    eri = np.asarray(ao2mo.restore(4, eri, norb), order="C")
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    assert fcivec.size == na * nb
    assert eri.dtype == np.float64
    if fcivec.dtype == eri.dtype == numpy.float64:
        fcivec = numpy.asarray(fcivec, order="C")
        eri = numpy.asarray(eri, order="C")
        ci1 = np.empty_like(fcivec)
        libfci.FCIcontract_2e_spin1(
            eri.ctypes.data_as(ctypes.c_void_p),
            fcivec.ctypes.data_as(ctypes.c_void_p),
            ci1.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(norb),
            ctypes.c_int(na),
            ctypes.c_int(nb),
            ctypes.c_int(nlinka),
            ctypes.c_int(nlinkb),
            link_indexa.ctypes.data_as(ctypes.c_void_p),
            link_indexb.ctypes.data_as(ctypes.c_void_p),
        )
        return ci1.view(FCIvector)

    ciR = numpy.asarray(fcivec.real, order="C")  # noqa: N806
    ciI = numpy.asarray(fcivec.imag, order="C")  # noqa: N806
    link_index = (link_indexa, link_indexb)
    outR = contract_2e(eri, ciR, norb, nelec, link_index=link_index)  # noqa: N806
    outI = contract_2e(eri, ciI, norb, nelec, link_index=link_index)  # noqa: N806
    out = outR.astype(numpy.complex128)
    out.imag = outI
    return out


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
    two_body_tensor = absorb_h1e(one_body_tensor, two_body_tensor, norb, nelec, 0.5)

    def matvec(vec: np.ndarray):
        result = contract_2e(
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
        result = contract_2e(
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
