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
from functools import cached_property

import numpy as np
import scipy.linalg
from opt_einsum import contract
from pyscf.fci.direct_nosym import make_hdiag
from pyscf.fci.direct_uhf import make_hdiag as make_hdiag_uhf
from scipy.sparse.linalg import LinearOperator

from ffsim import protocols
from ffsim.contract.two_body import two_body_linop
from ffsim.operators import FermionOperator, cre_a, cre_b, des_a, des_b


@dataclasses.dataclass(frozen=True)
class MolecularHamiltonian(
    protocols.SupportsApproximateEquality,
    protocols.SupportsDiagonal,
    protocols.SupportsFermionOperator,
    protocols.SupportsLinearOperator,
):
    r"""A molecular Hamiltonian.

    A Hamiltonian of the form

    .. math::

        H = \sum_{\substack{pq \\ \sigma}} h_{pq} a^\dagger_{p\sigma} a_{q\sigma}
            + \frac12 \sum_{\substack{pqrs \\ \sigma \tau}} h_{pqrs}
            a^\dagger_{p\sigma} a^\dagger_{r\tau} a_{s\tau} a_{q\sigma}
            + \text{constant}.

    Here :math:`h_{pq}` is called the one-body tensor and :math:`h_{pqrs}` is called
    the two-body tensor.
    """

    one_body_tensor: np.ndarray
    """The one-body tensor."""
    two_body_tensor: np.ndarray
    """The two-body tensor."""
    constant: float = 0.0
    """The constant."""

    @property
    def norb(self) -> int:
        """The number of spatial orbitals."""
        return self.one_body_tensor.shape[0]

    @cached_property
    def one_body_tensor_spinless(self) -> np.ndarray:
        """The one-body tensor in spinless format."""
        return self.to_spinless().one_body_tensor

    @cached_property
    def two_body_tensor_spinless(self) -> np.ndarray:
        """The two-body tensor in spinless format."""
        return self.to_spinless().two_body_tensor

    def to_spinless(self) -> MolecularHamiltonianSpinless:
        """Convert to a spinless molecular Hamiltonian."""
        norb = self.norb
        one_body = scipy.linalg.block_diag(self.one_body_tensor, self.one_body_tensor)
        two_body = np.zeros(
            (2 * norb, 2 * norb, 2 * norb, 2 * norb), dtype=self.two_body_tensor.dtype
        )
        two_body[:norb, :norb, :norb, :norb] = self.two_body_tensor
        two_body[:norb, :norb, norb:, norb:] = self.two_body_tensor
        two_body[norb:, norb:, :norb, :norb] = self.two_body_tensor
        two_body[norb:, norb:, norb:, norb:] = self.two_body_tensor
        return MolecularHamiltonianSpinless(
            one_body_tensor=one_body, two_body_tensor=two_body, constant=self.constant
        )

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

    def to_unrestricted(self) -> MolecularHamiltonianUnrestricted:
        """Convert to a spin-unrestricted molecular Hamiltonian.

        Returns a :class:`MolecularHamiltonianUnrestricted` representing the same
        Hamiltonian, with the one- and two-body tensors replicated across the spin
        sectors.

        Returns:
            The spin-unrestricted molecular Hamiltonian.
        """
        return MolecularHamiltonianUnrestricted(
            one_body_tensors=np.stack([self.one_body_tensor] * 2),
            two_body_tensors=np.stack([self.two_body_tensor] * 3),
            constant=self.constant,
        )

    def _linear_operator_(
        self, norb: int, nelec: int | tuple[int, int]
    ) -> LinearOperator:
        """Return a SciPy LinearOperator representing the object."""
        assert isinstance(nelec, tuple)
        return two_body_linop(
            self.two_body_tensor,
            norb=norb,
            nelec=nelec,
            one_body_tensor=self.one_body_tensor,
            constant=self.constant,
        )

    def _diag_(self, norb: int, nelec: int | tuple[int, int]) -> np.ndarray:
        """Return the diagonal entries of the Hamiltonian."""
        assert isinstance(nelec, tuple)
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
        coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {(): self.constant}
        for p, q in itertools.product(range(self.norb), repeat=2):
            coeff = self.one_body_tensor[p, q]
            coeffs[cre_a(p), des_a(q)] = coeff
            coeffs[cre_b(p), des_b(q)] = coeff
        for p, q, r, s in itertools.product(range(self.norb), repeat=4):
            coeff = 0.5 * self.two_body_tensor[p, q, r, s]
            coeffs[cre_a(p), cre_a(r), des_a(s), des_a(q)] = coeff
            coeffs[cre_a(p), cre_b(r), des_b(s), des_a(q)] = coeff
            coeffs[cre_b(p), cre_a(r), des_a(s), des_b(q)] = coeff
            coeffs[cre_b(p), cre_b(r), des_b(s), des_b(q)] = coeff
        return FermionOperator(coeffs)

    @staticmethod
    def from_fermion_operator(op: FermionOperator) -> MolecularHamiltonian:
        r"""Initialize a MolecularHamiltonian from a FermionOperator.

        The input operator must contain only terms of the following form:

        - A real-valued constant
        - :math:`a^\dagger_{p\sigma} a_{q\sigma}`
        - :math:`a^\dagger_{p\sigma}a^\dagger_{r\tau}a_{s\tau}a_{q\sigma}`

        Any other terms will cause an error to be raised. No attempt will be made to
        normal-order terms.

        Args:
            op: The FermionOperator from which to initialize the MolecularHamiltonian.

        Returns:
            The MolecularHamiltonian represented by the input FermionOperator.
        """
        # extract number of spatial orbitals
        norb = 1 + max(orb for term in op for _, _, orb in term)

        # initialize constant, one- and two-body tensors
        constant: float = 0.0
        one_body_tensor = np.zeros((norb, norb), dtype=complex)
        two_body_tensor = np.zeros((norb, norb, norb, norb), dtype=complex)

        for term, coeff in op.items():
            # constant term
            if not term:
                if coeff.imag:
                    raise ValueError(
                        f"Constant term must be real. Instead, got {coeff}."
                    )
                constant = coeff.real
            # one-body term
            elif len(term) == 2:
                (_, _, p), (_, _, q) = term
                valid_one_body = [(cre_a(p), des_a(q)), (cre_b(p), des_b(q))]
                if term in valid_one_body:
                    one_body_tensor[p, q] += 0.5 * coeff
                else:
                    raise ValueError(
                        "FermionOperator cannot be converted to MolecularHamiltonian. "
                        f"The quadratic term {term} is not of the required form "
                        r"a^\dagger_{p\sigma} a_{q\sigma}."
                    )
            # two-body term
            elif len(term) == 4:
                (_, _, p), (_, _, r), (_, _, s), (_, _, q) = term
                valid_two_body = [
                    (cre_a(p), cre_a(r), des_a(s), des_a(q)),
                    (cre_a(p), cre_b(r), des_b(s), des_a(q)),
                    (cre_b(p), cre_a(r), des_a(s), des_b(q)),
                    (cre_b(p), cre_b(r), des_b(s), des_b(q)),
                ]
                if term not in valid_two_body:
                    raise ValueError(
                        "FermionOperator cannot be converted to MolecularHamiltonian. "
                        f"The quartic term {term} is not of the required form "
                        r"a^\dagger_{p\sigma}a^\dagger_{r\tau}a_{s\tau}a_{q\sigma}."
                    )
                two_body_tensor[p, q, r, s] += 0.5 * coeff
            # other terms
            else:
                raise ValueError(
                    "FermionOperator cannot be converted to MolecularHamiltonian."
                    f" The term {term} is neither a constant, one-body, nor two-body "
                    "term."
                )

        return MolecularHamiltonian(
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            constant=constant,
        )

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


@dataclasses.dataclass(frozen=True)
class MolecularHamiltonianSpinless(
    protocols.SupportsApproximateEquality,
    protocols.SupportsDiagonal,
    protocols.SupportsFermionOperator,
    protocols.SupportsLinearOperator,
):
    r"""A spinless molecular Hamiltonian.

    A Hamiltonian of the form

    .. math::

        H = \sum_{pq} h_{pq} a^\dagger_{p} a_{q}
            + \frac12 \sum_{pqrs} h_{pqrs}
            a^\dagger_{p} a^\dagger_{r} a_{s} a_{q}
            + \text{constant}.

    Here :math:`h_{pq}` is called the one-body tensor and :math:`h_{pqrs}` is called
    the two-body tensor.

    """

    one_body_tensor: np.ndarray
    """The one-body tensor."""
    two_body_tensor: np.ndarray
    """The two-body tensor."""
    constant: float = 0.0
    """The constant."""

    @property
    def norb(self) -> int:
        """The number of orbitals."""
        return self.one_body_tensor.shape[0]

    def rotated(self, orbital_rotation: np.ndarray) -> MolecularHamiltonianSpinless:
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
        return MolecularHamiltonianSpinless(
            one_body_tensor=one_body_tensor_rotated,
            two_body_tensor=two_body_tensor_rotated,
            constant=self.constant,
        )

    def _linear_operator_(
        self, norb: int, nelec: int | tuple[int, int]
    ) -> LinearOperator:
        """Return a SciPy LinearOperator representing the object."""
        assert isinstance(nelec, int)
        return two_body_linop(
            self.two_body_tensor,
            norb=norb,
            nelec=(nelec, 0),
            one_body_tensor=self.one_body_tensor,
            constant=self.constant,
        )

    def _diag_(self, norb: int, nelec: int | tuple[int, int]) -> np.ndarray:
        """Return the diagonal entries of the Hamiltonian."""
        assert isinstance(nelec, int)
        nelec = (nelec, 0)
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
            op += FermionOperator({(cre_a(p), des_a(q)): coeff})
        for p, q, r, s in itertools.product(range(self.norb), repeat=4):
            coeff = 0.5 * self.two_body_tensor[p, q, r, s]
            op += FermionOperator({(cre_a(p), cre_a(r), des_a(s), des_a(q)): coeff})
        return op

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        if isinstance(other, MolecularHamiltonianSpinless):
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


@dataclasses.dataclass(frozen=True)
class MolecularHamiltonianUnrestricted(
    protocols.SupportsApproximateEquality,
    protocols.SupportsDiagonal,
    protocols.SupportsFermionOperator,
    protocols.SupportsLinearOperator,
):
    r"""A spin-unrestricted molecular Hamiltonian.

    A Hamiltonian of the form

    .. math::

        H = \sum_{\substack{pq \\ \sigma}} h^{\sigma}_{pq}
            a^\dagger_{p\sigma} a_{q\sigma}
            + \frac12 \sum_{pqrs} h^{\alpha\alpha}_{pqrs}
            a^\dagger_{p\alpha} a^\dagger_{r\alpha} a_{s\alpha} a_{q\alpha}
            + \sum_{pqrs} h^{\alpha\beta}_{pqrs}
            a^\dagger_{p\alpha} a^\dagger_{r\beta} a_{s\beta} a_{q\alpha}
            + \frac12 \sum_{pqrs} h^{\beta\beta}_{pqrs}
            a^\dagger_{p\beta} a^\dagger_{r\beta} a_{s\beta} a_{q\beta}
            + \text{constant}.

    Here :math:`h^{\alpha}` and :math:`h^{\beta}` are the one-body tensors and
    :math:`h^{\alpha\alpha}`, :math:`h^{\alpha\beta}`, and :math:`h^{\beta\beta}` are
    the two-body tensors. The two-body tensors use chemist ordering, and their spin
    labels refer to the two *pairs* of indices: in :math:`h^{\alpha\beta}_{pqrs}`, the
    indices :math:`pq` belong to spin alpha and the indices :math:`rs` belong to spin
    beta.

    Unlike :class:`MolecularHamiltonian`, which uses a single one-body tensor and a
    single two-body tensor shared by both spin sectors, this class stores independent
    tensors for each spin sector. As a result, it can represent Hamiltonians that are
    not invariant under exchange of the spin alpha and spin beta orbitals, such as the
    result of applying a different orbital rotation to each spin sector. See
    :meth:`rotated`.

    The beta-alpha two-body tensor is not stored because it is determined by the
    alpha-beta tensor:

    .. math::

        h^{\beta\alpha}_{pqrs} = h^{\alpha\beta}_{rspq}.

    Note that the alpha-beta term in the Hamiltonian above therefore appears with
    coefficient 1 rather than :math:`\frac12`, since the implicit beta-alpha term
    contributes the other half.
    """

    one_body_tensors: np.ndarray
    r"""The one-body tensors :math:`(h^{\alpha}, h^{\beta})`, as a single Numpy array
    of shape ``(2, norb, norb)``."""
    two_body_tensors: np.ndarray
    r"""The two-body tensors
    :math:`(h^{\alpha\alpha}, h^{\alpha\beta}, h^{\beta\beta})`, as a single Numpy
    array of shape ``(3, norb, norb, norb, norb)``."""
    constant: float = 0.0
    """The constant."""

    @property
    def norb(self) -> int:
        """The number of spatial orbitals."""
        return self.one_body_tensors.shape[1]

    def rotated(
        self,
        orbital_rotation: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
    ) -> MolecularHamiltonianUnrestricted:
        r"""Return the Hamiltonian in a rotated orbital basis.

        Given an orbital rotation :math:`\mathcal{U}`, returns the operator

        .. math::

            \mathcal{U} H \mathcal{U}^\dagger

        where :math:`H` is the original Hamiltonian.

        Args:
            orbital_rotation: The orbital rotation.
                You can pass either a single Numpy array specifying the orbital
                rotation to apply to both spin sectors, or you can pass a pair of Numpy
                arrays specifying independent orbital rotations for spin alpha and spin
                beta. If passing a pair, you can use ``None`` for one of the values in
                the pair to indicate that no operation should be applied to that spin
                sector.

        Returns:
            The rotated Hamiltonian.
        """
        rotation_a, rotation_b = _unpack_orbital_rotation(orbital_rotation)
        return MolecularHamiltonianUnrestricted(
            one_body_tensors=np.stack(
                [
                    _rotate_one_body(self.one_body_tensors[0], rotation_a),
                    _rotate_one_body(self.one_body_tensors[1], rotation_b),
                ]
            ),
            two_body_tensors=np.stack(
                [
                    _rotate_two_body(self.two_body_tensors[0], rotation_a, rotation_a),
                    _rotate_two_body(self.two_body_tensors[1], rotation_a, rotation_b),
                    _rotate_two_body(self.two_body_tensors[2], rotation_b, rotation_b),
                ]
            ),
            constant=self.constant,
        )

    def to_spinless(self) -> MolecularHamiltonianSpinless:
        r"""Convert to a spinless molecular Hamiltonian.

        Returns a :class:`MolecularHamiltonianSpinless` on ``2 * norb`` orbitals
        representing the same Hamiltonian, with the spin alpha orbitals occupying the
        first ``norb`` orbitals and the spin beta orbitals the last ``norb`` orbitals.

        Returns:
            The spinless molecular Hamiltonian.
        """
        norb = self.norb
        one_body_a, one_body_b = self.one_body_tensors
        two_body_aa, two_body_ab, two_body_bb = self.two_body_tensors

        one_body_tensor = scipy.linalg.block_diag(one_body_a, one_body_b)
        two_body_tensor = np.zeros((2 * norb,) * 4, dtype=self.two_body_tensors.dtype)
        # The spinless two-body operator is defined with a factor of 1/2 and sums over
        # both orderings of the index pairs, so the same-spin blocks are symmetrized
        # under exchange of the pairs. Both orderings define the same operator, so this
        # does not change the Hamiltonian.
        two_body_tensor[:norb, :norb, :norb, :norb] = 0.5 * (
            two_body_aa + two_body_aa.transpose(2, 3, 0, 1)
        )
        two_body_tensor[norb:, norb:, norb:, norb:] = 0.5 * (
            two_body_bb + two_body_bb.transpose(2, 3, 0, 1)
        )
        # The alpha-beta tensor need not be symmetric under pair exchange, and must not
        # be symmetrized: it and its transpose fill the two off-diagonal blocks, which
        # reproduces the alpha-beta and beta-alpha terms of the Hamiltonian separately.
        two_body_tensor[:norb, :norb, norb:, norb:] = two_body_ab
        two_body_tensor[norb:, norb:, :norb, :norb] = two_body_ab.transpose(2, 3, 0, 1)

        return MolecularHamiltonianSpinless(
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            constant=self.constant,
        )

    def _linear_operator_(
        self, norb: int, nelec: int | tuple[int, int]
    ) -> LinearOperator:
        """Return a SciPy LinearOperator representing the object."""
        assert isinstance(nelec, tuple)
        return protocols.linear_operator(
            self._fermion_operator_(), norb=norb, nelec=nelec
        )

    def _diag_(self, norb: int, nelec: int | tuple[int, int]) -> np.ndarray:
        """Return the diagonal entries of the Hamiltonian."""
        assert isinstance(nelec, tuple)
        if np.iscomplexobj(self.one_body_tensors) or np.iscomplexobj(
            self.two_body_tensors
        ):
            raise NotImplementedError(
                "Computing diagonal of complex molecular Hamiltonian is not yet "
                "supported."
            )
        return (
            make_hdiag_uhf(
                (self.one_body_tensors[0], self.one_body_tensors[1]),
                (
                    self.two_body_tensors[0],
                    self.two_body_tensors[1],
                    self.two_body_tensors[2],
                ),
                norb,
                nelec,
            )
            + self.constant
        )

    def _fermion_operator_(self) -> FermionOperator:
        """Return a FermionOperator representing the object."""
        one_body_a, one_body_b = self.one_body_tensors
        two_body_aa, two_body_ab, two_body_bb = self.two_body_tensors
        coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {(): self.constant}
        for p, q in itertools.product(range(self.norb), repeat=2):
            coeffs[cre_a(p), des_a(q)] = one_body_a[p, q]
            coeffs[cre_b(p), des_b(q)] = one_body_b[p, q]
        for p, q, r, s in itertools.product(range(self.norb), repeat=4):
            coeffs[cre_a(p), cre_a(r), des_a(s), des_a(q)] = (
                0.5 * two_body_aa[p, q, r, s]
            )
            coeffs[cre_b(p), cre_b(r), des_b(s), des_b(q)] = (
                0.5 * two_body_bb[p, q, r, s]
            )
            coeffs[cre_a(p), cre_b(r), des_b(s), des_a(q)] = (
                0.5 * two_body_ab[p, q, r, s]
            )
            # The beta-alpha tensor is the transpose of the alpha-beta tensor.
            coeffs[cre_b(p), cre_a(r), des_a(s), des_b(q)] = (
                0.5 * two_body_ab[r, s, p, q]
            )
        return FermionOperator(coeffs)

    @staticmethod
    def from_fermion_operator(
        op: FermionOperator,
    ) -> MolecularHamiltonianUnrestricted:
        r"""Initialize a MolecularHamiltonianUnrestricted from a FermionOperator.

        The input operator must contain only terms of the following form:

        - A real-valued constant
        - :math:`a^\dagger_{p\sigma} a_{q\sigma}`
        - :math:`a^\dagger_{p\sigma}a^\dagger_{r\tau}a_{s\tau}a_{q\sigma}`

        Any other terms will cause an error to be raised. No attempt will be made to
        normal-order terms.

        Unlike :meth:`MolecularHamiltonian.from_fermion_operator`, the coefficients of
        the spin alpha and spin beta terms are stored independently rather than
        averaged, so spin-dependent operators are represented exactly.

        Args:
            op: The FermionOperator from which to initialize the Hamiltonian.

        Returns:
            The MolecularHamiltonianUnrestricted represented by the input
            FermionOperator.
        """
        # extract number of spatial orbitals
        norb = 1 + max(orb for term in op for _, _, orb in term)

        # initialize constant, one- and two-body tensors
        constant: float = 0.0
        one_body_tensors = np.zeros((2, norb, norb), dtype=complex)
        two_body_tensors = np.zeros((3, norb, norb, norb, norb), dtype=complex)

        for term, coeff in op.items():
            # constant term
            if not term:
                if coeff.imag:
                    raise ValueError(
                        f"Constant term must be real. Instead, got {coeff}."
                    )
                constant = coeff.real
            # one-body term
            elif len(term) == 2:
                (_, _, p), (_, _, q) = term
                if term == (cre_a(p), des_a(q)):
                    one_body_tensors[0, p, q] += coeff
                elif term == (cre_b(p), des_b(q)):
                    one_body_tensors[1, p, q] += coeff
                else:
                    raise ValueError(
                        "FermionOperator cannot be converted to "
                        "MolecularHamiltonianUnrestricted. The quadratic term "
                        f"{term} is not of the required form "
                        r"a^\dagger_{p\sigma} a_{q\sigma}."
                    )
            # two-body term
            elif len(term) == 4:
                (_, _, p), (_, _, r), (_, _, s), (_, _, q) = term
                if term == (cre_a(p), cre_a(r), des_a(s), des_a(q)):
                    two_body_tensors[0, p, q, r, s] += 2 * coeff
                elif term == (cre_b(p), cre_b(r), des_b(s), des_b(q)):
                    two_body_tensors[2, p, q, r, s] += 2 * coeff
                # The alpha-beta and beta-alpha terms both contribute to the alpha-beta
                # tensor, so each contributes half as much as a same-spin term does.
                elif term == (cre_a(p), cre_b(r), des_b(s), des_a(q)):
                    two_body_tensors[1, p, q, r, s] += coeff
                elif term == (cre_b(p), cre_a(r), des_a(s), des_b(q)):
                    two_body_tensors[1, r, s, p, q] += coeff
                else:
                    raise ValueError(
                        "FermionOperator cannot be converted to "
                        "MolecularHamiltonianUnrestricted. The quartic term "
                        f"{term} is not of the required form "
                        r"a^\dagger_{p\sigma}a^\dagger_{r\tau}a_{s\tau}a_{q\sigma}."
                    )
            # other terms
            else:
                raise ValueError(
                    "FermionOperator cannot be converted to "
                    f"MolecularHamiltonianUnrestricted. The term {term} is neither a "
                    "constant, one-body, nor two-body term."
                )

        return MolecularHamiltonianUnrestricted(
            one_body_tensors=one_body_tensors,
            two_body_tensors=two_body_tensors,
            constant=constant,
        )

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        if isinstance(other, MolecularHamiltonianUnrestricted):
            if not np.allclose(self.constant, other.constant, rtol=rtol, atol=atol):
                return False
            if not np.allclose(
                self.one_body_tensors, other.one_body_tensors, rtol=rtol, atol=atol
            ):
                return False
            if not np.allclose(
                self.two_body_tensors, other.two_body_tensors, rtol=rtol, atol=atol
            ):
                return False
            return True
        return NotImplemented


def _unpack_orbital_rotation(
    orbital_rotation: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Unpack an orbital rotation into separate alpha and beta orbital rotations."""
    if isinstance(orbital_rotation, np.ndarray) and orbital_rotation.ndim == 2:
        return orbital_rotation, orbital_rotation
    orbital_rotation_a, orbital_rotation_b = orbital_rotation
    return orbital_rotation_a, orbital_rotation_b


def _rotate_one_body(
    tensor: np.ndarray, orbital_rotation: np.ndarray | None
) -> np.ndarray:
    """Rotate a one-body tensor."""
    if orbital_rotation is None:
        return tensor
    return contract(
        "ab,Aa,Bb->AB",
        tensor,
        orbital_rotation,
        orbital_rotation.conj(),
        optimize="greedy",
    )


def _rotate_two_body(
    tensor: np.ndarray,
    orbital_rotation_1: np.ndarray | None,
    orbital_rotation_2: np.ndarray | None,
) -> np.ndarray:
    """Rotate a two-body tensor.

    The first orbital rotation acts on the first pair of indices and the second orbital
    rotation acts on the second pair of indices.
    """
    if orbital_rotation_1 is None and orbital_rotation_2 is None:
        return tensor
    norb = tensor.shape[0]
    if orbital_rotation_1 is None:
        orbital_rotation_1 = np.eye(norb)
    if orbital_rotation_2 is None:
        orbital_rotation_2 = np.eye(norb)
    return contract(
        "abcd,Aa,Bb,Cc,Dd->ABCD",
        tensor,
        orbital_rotation_1,
        orbital_rotation_1.conj(),
        orbital_rotation_2,
        orbital_rotation_2.conj(),
        optimize="greedy",
    )
