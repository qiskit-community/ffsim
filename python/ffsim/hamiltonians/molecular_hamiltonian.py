# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import dataclasses
import itertools
import os

import numpy as np
import pyscf.ao2mo
import pyscf.tools
from opt_einsum import contract
from pyscf.fci.direct_nosym import absorb_h1e, contract_1e, contract_2e, make_hdiag
from scipy.sparse.linalg import LinearOperator
from typing_extensions import deprecated

from ffsim.cistring import gen_linkstr_index
from ffsim.operators import FermionOperator, cre_a, cre_b, des_a, des_b
from ffsim.states import dim


@dataclasses.dataclass(frozen=True)
class MolecularHamiltonian:
    r"""A molecular Hamiltonian.

    A Hamiltonian of the form

    .. math::

        H = \sum_{\sigma, pq} h_{pq} a^\dagger_{\sigma, p} a_{\sigma, q}
            + \frac12 \sum_{\sigma \tau, pqrs} h_{pqrs}
            a^\dagger_{\sigma, p} a^\dagger_{\tau, r} a_{\tau, s} a_{\sigma, q}
            + \text{constant}.

    Here :math:`h_{pq}` is called the one-body tensor and :math:`h_{pqrs}` is called
    the two-body tensor.

    Attributes:
        one_body_tensor (np.ndarray): The one-body tensor.
        two_body_tensor (np.ndarray): The two-body tensor.
        constant (float): The constant.
    """

    one_body_tensor: np.ndarray
    two_body_tensor: np.ndarray
    constant: float = 0.0

    @staticmethod
    @deprecated(
        "The MolecularHamiltonian.from_fcidump method is deprecated. "
        "Instead, use MolecularData.from_fcidump and then access the `hamiltonian` "
        "attribute of the returned MolecularData."
    )
    def from_fcidump(file: str | bytes | os.PathLike) -> MolecularHamiltonian:
        """Initialize a MolecularHamiltonian from an FCIDUMP file.

        .. warning::
            This function is deprecated. Instead, use MolecularData.from_fcidump and
            then access the `hamiltonian` attribute of the returned MolecularData.

        Args:
            file: The FCIDUMP file path.
        """
        data = pyscf.tools.fcidump.read(file, verbose=False)
        return MolecularHamiltonian(
            one_body_tensor=data["H1"],
            two_body_tensor=pyscf.ao2mo.restore(1, data["H2"], data["NORB"]),
            constant=data["ECORE"],
        )

    @property
    def norb(self) -> int:
        """The number of spatial orbitals."""
        return self.one_body_tensor.shape[0]

    def rotated(self, orbital_rotation: np.ndarray) -> MolecularHamiltonian:
        r"""Return the Hamiltonian in a rotated orbital basis.

        Given an orbital rotation :math:`\mathcal{U}`, returns the operator

        .. math::

            \mathcal{U} H \mathcal{U}^\dagger

        where :math:`H` is the original Hamiltonian.

        Args:
            orbital_rotation: The orbital rotation.

        Returns:
            The rotated Hamiltonian.
        """
        one_body_tensor_rotated = contract(
            "ab,Aa,Bb->AB",
            self.one_body_tensor,
            orbital_rotation,
            orbital_rotation.conj(),
            optimize="greedy",
        )
        two_body_tensor_rotated = contract(
            "abcd,Aa,Bb,Cc,Dd->ABCD",
            self.two_body_tensor,
            orbital_rotation,
            orbital_rotation.conj(),
            orbital_rotation,
            orbital_rotation.conj(),
            optimize="greedy",
        )
        return MolecularHamiltonian(
            one_body_tensor=one_body_tensor_rotated,
            two_body_tensor=two_body_tensor_rotated,
            constant=self.constant,
        )

    def _linear_operator_(self, norb: int, nelec: tuple[int, int]) -> LinearOperator:
        """Return a SciPy LinearOperator representing the object."""
        if np.iscomplexobj(self.two_body_tensor):
            raise NotImplementedError(
                "This Hamiltonian has a complex-valued two-body tensor. "
                "LinearOperator support for complex two-body tensors is not yet "
                "implemented. See https://github.com/qiskit-community/ffsim/issues/81."
            )

        n_alpha, n_beta = nelec
        linkstr_index_a = gen_linkstr_index(range(norb), n_alpha)
        linkstr_index_b = gen_linkstr_index(range(norb), n_beta)
        link_index = (linkstr_index_a, linkstr_index_b)

        if np.iscomplexobj(self.one_body_tensor):
            return _linear_operator_complex(
                one_body_tensor=self.one_body_tensor,
                two_body_tensor=self.two_body_tensor,
                constant=self.constant,
                norb=norb,
                nelec=nelec,
                link_index=link_index,
            )
        return _linear_operator_real(
            one_body_tensor=self.one_body_tensor,
            two_body_tensor=self.two_body_tensor,
            constant=self.constant,
            norb=norb,
            nelec=nelec,
            link_index=link_index,
        )

    def _diag_(self, norb: int, nelec: tuple[int, int]) -> np.ndarray:
        """Return the diagonal entries of the Hamiltonian."""
        if np.iscomplexobj(self.two_body_tensor) or np.iscomplexobj(
            self.one_body_tensor
        ):
            raise NotImplementedError(
                "Computing diagonal of complex molecular Hamiltonian is not yet "
                "supported."
            )
        return (
            make_hdiag(self.one_body_tensor, self.two_body_tensor, norb, nelec)
            + self.constant
        )

    def _fermion_operator_(self) -> FermionOperator:
        """Return a FermionOperator representing the object."""
        op = FermionOperator({(): self.constant})
        for p, q in itertools.product(range(self.norb), repeat=2):
            coeff = self.one_body_tensor[p, q]
            op += FermionOperator(
                {
                    (cre_a(p), des_a(q)): coeff,
                    (cre_b(p), des_b(q)): coeff,
                }
            )
        for p, q, r, s in itertools.product(range(self.norb), repeat=4):
            coeff = 0.5 * self.two_body_tensor[p, q, r, s]
            op += FermionOperator(
                {
                    (cre_a(p), cre_a(r), des_a(s), des_a(q)): coeff,
                    (cre_a(p), cre_b(r), des_b(s), des_a(q)): coeff,
                    (cre_b(p), cre_a(r), des_a(s), des_b(q)): coeff,
                    (cre_b(p), cre_b(r), des_b(s), des_b(q)): coeff,
                }
            )
        return op

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        if isinstance(other, MolecularHamiltonian):
            if not np.allclose(self.constant, other.constant, rtol=rtol, atol=atol):
                return False
            if not np.allclose(
                self.one_body_tensor, other.one_body_tensor, rtol=rtol, atol=atol
            ):
                return False
            if not np.allclose(
                self.two_body_tensor, other.two_body_tensor, rtol=rtol, atol=atol
            ):
                return False
            return True
        return NotImplemented


def _linear_operator_complex(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    constant: float,
    norb: int,
    nelec: tuple[int, int],
    link_index: tuple[np.ndarray, np.ndarray],
):
    two_body = absorb_h1e(one_body_tensor.real, two_body_tensor, norb, nelec, 0.5)
    dim_ = dim(norb, nelec)

    def matvec(vec: np.ndarray):
        result = constant * vec.astype(complex, copy=False)
        result += 1j * contract_1e(
            one_body_tensor.imag, vec.real, norb, nelec, link_index=link_index
        )
        result -= contract_1e(
            one_body_tensor.imag, vec.imag, norb, nelec, link_index=link_index
        )
        result += contract_2e(two_body, vec.real, norb, nelec, link_index=link_index)
        result += 1j * contract_2e(
            two_body, vec.imag, norb, nelec, link_index=link_index
        )
        return result

    return LinearOperator(
        shape=(dim_, dim_), matvec=matvec, rmatvec=matvec, dtype=complex
    )


def _linear_operator_real(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    constant: float,
    norb: int,
    nelec: tuple[int, int],
    link_index: tuple[np.ndarray, np.ndarray],
):
    two_body = absorb_h1e(one_body_tensor, two_body_tensor, norb, nelec, 0.5)
    dim_ = dim(norb, nelec)

    def matvec(vec: np.ndarray):
        result = constant * vec.astype(complex, copy=False)
        result += contract_2e(two_body, vec.real, norb, nelec, link_index=link_index)
        result += 1j * contract_2e(
            two_body, vec.imag, norb, nelec, link_index=link_index
        )
        return result

    return LinearOperator(
        shape=(dim_, dim_), matvec=matvec, rmatvec=matvec, dtype=complex
    )
