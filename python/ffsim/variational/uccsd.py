# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unitary coupled cluster, singles and doubles ansatz."""

from __future__ import annotations

import itertools
from dataclasses import InitVar, dataclass
from typing import cast

import numpy as np
import scipy.sparse.linalg

from ffsim import contract, gates, linalg, operators, protocols
from ffsim.linalg.util import unitary_from_parameters, unitary_to_parameters


def uccsd_restricted_linear_operator(
    t1: np.ndarray, t2: np.ndarray, norb: int, nelec: tuple[int, int]
) -> scipy.sparse.linalg.LinearOperator:
    """Return a linear operator for a restricted UCCSD operator generator.

    Args:
        t1: The t1-amplitudes.
        t2: The t2-amplitudes.
        norb: The number of spatial orbitals.
        nelec: The numbers of spin alpha and spin beta fermions.

    Returns:
        The LinearOperator for the UCCSD operator generator.
    """
    nocc, _ = t1.shape
    assert nelec == (nocc, nocc)

    one_body_tensor = np.zeros((norb, norb), dtype=complex)
    two_body_tensor = np.zeros((norb, norb, norb, norb), dtype=complex)
    one_body_tensor[:nocc, nocc:] = -t1.conj()
    one_body_tensor[nocc:, :nocc] = t1.T
    two_body_tensor[nocc:, :nocc, nocc:, :nocc] = t2.transpose(2, 0, 3, 1)
    two_body_tensor[:nocc, nocc:, :nocc, nocc:] = -t2.transpose(0, 2, 1, 3).conj()
    return contract.two_body_linop(
        two_body_tensor, norb=norb, nelec=nelec, one_body_tensor=one_body_tensor
    )


def uccsd_unrestricted_linear_operator(
    t1: tuple[np.ndarray, np.ndarray],
    t2: tuple[np.ndarray, np.ndarray, np.ndarray],
    norb: int,
    nelec: tuple[int, int],
) -> scipy.sparse.linalg.LinearOperator:
    """Return a linear operator for an unrestricted UCCSD operator generator.

    Args:
        t1: The t1-amplitudes.
        t2: The t2-amplitudes.
        norb: The number of spatial orbitals.
        nelec: The numbers of spin alpha and spin beta fermions.

    Returns:
        The LinearOperator for the UCCSD operator generator.
    """
    return protocols.linear_operator(
        operators.uccsd_generator_unrestricted(t1, t2), norb=norb, nelec=nelec
    )


@dataclass(frozen=True)
class UCCSDOpRestrictedReal(
    protocols.SupportsApplyUnitary, protocols.SupportsApproximateEquality
):
    """Real-valued restricted unitary coupled cluster, singles and doubles operator.

    UCCSD operator with real-valued t-amplitudes. Note that the final orbital rotation,
    if included, is allowed to be complex-valued.
    """

    t1: np.ndarray  # shape: (nocc, nvrt)
    t2: np.ndarray  # shape: (nocc, nocc, nvrt, nvrt)
    final_orbital_rotation: np.ndarray | None = None  # shape: (norb, norb)
    validate: InitVar[bool] = True
    rtol: InitVar[float] = 1e-5
    atol: InitVar[float] = 1e-8

    def __post_init__(self, validate: bool, rtol: float, atol: float):
        if np.iscomplexobj(self.t1):
            raise TypeError(
                "UCCSDOpRestricted only accepts real-valued t1 amplitudes. "
                "Please pass a t1 amplitudes tensor with a real-valued data type."
            )
        if np.iscomplexobj(self.t2):
            raise TypeError(
                "UCCSDOpRestricted only accepts real-valued t2 amplitudes. "
                "Please pass a t2 amplitudes tensor with a real-valued data type."
            )
        if validate:
            # Validate shapes
            nocc, nvrt = self.t1.shape
            norb = nocc + nvrt
            if self.t2.shape != (nocc, nocc, nvrt, nvrt):
                raise ValueError(
                    "t2 shape not consistent with t1 shape. "
                    f"Expected {(nocc, nocc, nvrt, nvrt)} but got {self.t2.shape}"
                )
            if self.final_orbital_rotation is not None:
                if self.final_orbital_rotation.shape != (norb, norb):
                    raise ValueError(
                        "Final orbital rotation shape not consistent with t1 shape. "
                        f"Expected {(norb, norb)} but got "
                        f"{self.final_orbital_rotation.shape}"
                    )
                if not linalg.is_unitary(
                    self.final_orbital_rotation, rtol=rtol, atol=atol
                ):
                    raise ValueError("Final orbital rotation was not unitary.")

    @property
    def norb(self):
        """The number of spatial orbitals."""
        nocc, nvrt = self.t1.shape
        return nocc + nvrt

    @staticmethod
    def n_params(
        norb: int, nocc: int, *, with_final_orbital_rotation: bool = False
    ) -> int:
        """Return the number of parameters of an ansatz with given settings.

        Args:
            norb: The number of spatial orbitals.
            nocc: The number of spatial orbitals that are occupied by electrons.
            with_final_orbital_rotation: Whether to include a final orbital rotation
                in the operator.

        Returns:
            The number of parameters of the ansatz.
        """
        nvrt = norb - nocc
        # Number of occupied-virtual pairs
        n_pairs = nocc * nvrt
        # t1 has n_pairs parameters
        # t2 has n_pairs * (n_pairs + 1) // 2 parameters
        # Final orbital rotation has norb**2 parameters
        return (
            n_pairs
            + n_pairs * (n_pairs + 1) // 2
            + with_final_orbital_rotation * norb**2
        )

    @staticmethod
    def from_parameters(
        params: np.ndarray,
        *,
        norb: int,
        nocc: int,
        with_final_orbital_rotation: bool = False,
    ) -> UCCSDOpRestrictedReal:
        r"""Initialize the UCCSD operator from a real-valued parameter vector.

        Args:
            params: The real-valued parameter vector.
            norb: The number of spatial orbitals.
            nocc: The number of spatial orbitals that are occupied by electrons.
            with_final_orbital_rotation: Whether to include a final orbital rotation
                in the operator.

        Returns:
            The UCCSD operator constructed from the given parameters.

        Raises:
            ValueError: The number of parameters passed did not match the number
                expected based on the function inputs.
        """
        n_params = UCCSDOpRestrictedReal.n_params(
            norb, nocc, with_final_orbital_rotation=with_final_orbital_rotation
        )
        if len(params) != n_params:
            raise ValueError(
                "The number of parameters passed did not match the number expected "
                "based on the function inputs. "
                f"Expected {n_params} but got {len(params)}."
            )
        nvrt = norb - nocc
        t1 = np.zeros((nocc, nvrt))
        t2 = np.zeros((nocc, nocc, nvrt, nvrt))
        occ_vrt_pairs = list(itertools.product(range(nocc), range(nocc, norb)))
        index = 0
        # t1
        for i, a in occ_vrt_pairs:
            t1[i, a - nocc] = params[index]
            index += 1
        # t2
        for (i, a), (j, b) in itertools.combinations_with_replacement(occ_vrt_pairs, 2):
            t2[i, j, a - nocc, b - nocc] = params[index]
            t2[j, i, b - nocc, a - nocc] = params[index]
            index += 1
        # Final orbital rotation
        final_orbital_rotation = None
        if with_final_orbital_rotation:
            final_orbital_rotation = unitary_from_parameters(params[index:], norb)
        return UCCSDOpRestrictedReal(
            t1=t1, t2=t2, final_orbital_rotation=final_orbital_rotation
        )

    def to_parameters(self) -> np.ndarray:
        r"""Convert the UCCSD operator to a real-valued parameter vector.

        Returns:
            The real-valued parameter vector.
        """
        nocc, nvrt = self.t1.shape
        norb = nocc + nvrt
        n_params = UCCSDOpRestrictedReal.n_params(
            norb,
            nocc,
            with_final_orbital_rotation=self.final_orbital_rotation is not None,
        )
        params = np.zeros(n_params)
        occ_vrt_pairs = list(itertools.product(range(nocc), range(nocc, norb)))
        index = 0
        # t1
        for i, a in occ_vrt_pairs:
            params[index] = self.t1[i, a - nocc]
            index += 1
        # t2
        for (i, a), (j, b) in itertools.combinations_with_replacement(occ_vrt_pairs, 2):
            params[index] = self.t2[i, j, a - nocc, b - nocc]
            index += 1
        # Final orbital rotation
        if self.final_orbital_rotation is not None:
            params[index:] = unitary_to_parameters(self.final_orbital_rotation)
        return params

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        if isinstance(nelec, int):
            return NotImplemented
        if copy:
            vec = vec.copy()

        linop = uccsd_restricted_linear_operator(
            self.t1, self.t2, norb=norb, nelec=nelec
        )
        vec = scipy.sparse.linalg.expm_multiply(linop, vec, traceA=0.0)

        if self.final_orbital_rotation is not None:
            vec = gates.apply_orbital_rotation(
                vec, self.final_orbital_rotation, norb=norb, nelec=nelec, copy=False
            )

        return vec

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        if isinstance(other, UCCSDOpRestrictedReal):
            if not np.allclose(self.t1, other.t1, rtol=rtol, atol=atol):
                return False
            if not np.allclose(self.t2, other.t2, rtol=rtol, atol=atol):
                return False
            if (self.final_orbital_rotation is None) != (
                other.final_orbital_rotation is None
            ):
                return False
            if self.final_orbital_rotation is not None:
                return np.allclose(
                    cast(np.ndarray, self.final_orbital_rotation),
                    cast(np.ndarray, other.final_orbital_rotation),
                    rtol=rtol,
                    atol=atol,
                )
            return True
        return NotImplemented


@dataclass(frozen=True)
class UCCSDOpRestricted(
    protocols.SupportsApplyUnitary, protocols.SupportsApproximateEquality
):
    """Restricted unitary coupled cluster, singles and doubles operator.

    UCCSD operator with complex-valued t-amplitudes.
    """

    t1: np.ndarray  # shape: (nocc, nvrt)
    t2: np.ndarray  # shape: (nocc, nocc, nvrt, nvrt)
    final_orbital_rotation: np.ndarray | None = None  # shape: (norb, norb)
    validate: InitVar[bool] = True
    rtol: InitVar[float] = 1e-5
    atol: InitVar[float] = 1e-8

    def __post_init__(self, validate: bool, rtol: float, atol: float):
        if validate:
            # Validate shapes
            nocc, nvrt = self.t1.shape
            norb = nocc + nvrt
            if self.t2.shape != (nocc, nocc, nvrt, nvrt):
                raise ValueError(
                    "t2 shape not consistent with t1 shape. "
                    f"Expected {(nocc, nocc, nvrt, nvrt)} but got {self.t2.shape}"
                )
            if self.final_orbital_rotation is not None:
                if self.final_orbital_rotation.shape != (norb, norb):
                    raise ValueError(
                        "Final orbital rotation shape not consistent with t1 shape. "
                        f"Expected {(norb, norb)} but got "
                        f"{self.final_orbital_rotation.shape}"
                    )
                if not linalg.is_unitary(
                    self.final_orbital_rotation, rtol=rtol, atol=atol
                ):
                    raise ValueError("Final orbital rotation was not unitary.")

    @property
    def norb(self):
        """The number of spatial orbitals."""
        nocc, nvrt = self.t1.shape
        return nocc + nvrt

    @staticmethod
    def n_params(
        norb: int, nocc: int, *, with_final_orbital_rotation: bool = False
    ) -> int:
        """Return the number of parameters of an ansatz with given settings.

        Args:
            norb: The number of spatial orbitals.
            nocc: The number of spatial orbitals that are occupied by electrons.
            with_final_orbital_rotation: Whether to include a final orbital rotation
                in the operator.

        Returns:
            The number of parameters of the ansatz.
        """
        nvrt = norb - nocc
        # Number of occupied-virtual pairs
        n_pairs = nocc * nvrt
        # t1 has 2* n_pairs parameters
        # t2 has n_pairs * (n_pairs + 1) parameters
        # Final orbital rotation has norb**2 parameters
        return (
            2 * n_pairs
            + n_pairs * (n_pairs + 1)
            + with_final_orbital_rotation * norb**2
        )

    @staticmethod
    def from_parameters(
        params: np.ndarray,
        *,
        norb: int,
        nocc: int,
        with_final_orbital_rotation: bool = False,
    ) -> UCCSDOpRestricted:
        r"""Initialize the UCCSD operator from a real-valued parameter vector.

        Args:
            params: The real-valued parameter vector.
            norb: The number of spatial orbitals.
            nocc: The number of spatial orbitals that are occupied by electrons.
            with_final_orbital_rotation: Whether to include a final orbital rotation
                in the operator.

        Returns:
            The UCCSD operator constructed from the given parameters.

        Raises:
            ValueError: The number of parameters passed did not match the number
                expected based on the function inputs.
        """
        n_params = UCCSDOpRestricted.n_params(
            norb, nocc, with_final_orbital_rotation=with_final_orbital_rotation
        )
        if len(params) != n_params:
            raise ValueError(
                "The number of parameters passed did not match the number expected "
                "based on the function inputs. "
                f"Expected {n_params} but got {len(params)}."
            )
        nvrt = norb - nocc
        t1 = np.zeros((nocc, nvrt), dtype=complex)
        t2 = np.zeros((nocc, nocc, nvrt, nvrt), dtype=complex)
        occ_vrt_pairs = list(itertools.product(range(nocc), range(nocc, norb)))
        index = 0
        # t1
        for i, a in occ_vrt_pairs:
            t1[i, a - nocc] = params[index] + 1j * params[index + 1]
            index += 2
        # t2
        for (i, a), (j, b) in itertools.combinations_with_replacement(occ_vrt_pairs, 2):
            t2[i, j, a - nocc, b - nocc] = params[index] + 1j * params[index + 1]
            t2[j, i, b - nocc, a - nocc] = params[index] + 1j * params[index + 1]
            index += 2
        # Final orbital rotation
        final_orbital_rotation = None
        if with_final_orbital_rotation:
            final_orbital_rotation = unitary_from_parameters(params[index:], norb)
        return UCCSDOpRestricted(
            t1=t1, t2=t2, final_orbital_rotation=final_orbital_rotation
        )

    def to_parameters(self) -> np.ndarray:
        r"""Convert the UCCSD operator to a real-valued parameter vector.

        Returns:
            The real-valued parameter vector.
        """
        nocc, nvrt = self.t1.shape
        norb = nocc + nvrt
        n_params = UCCSDOpRestricted.n_params(
            norb,
            nocc,
            with_final_orbital_rotation=self.final_orbital_rotation is not None,
        )
        params = np.zeros(n_params)
        occ_vrt_pairs = list(itertools.product(range(nocc), range(nocc, norb)))
        index = 0
        # t1
        for i, a in occ_vrt_pairs:
            params[index] = self.t1[i, a - nocc].real
            params[index + 1] = self.t1[i, a - nocc].imag
            index += 2
        # t2
        for (i, a), (j, b) in itertools.combinations_with_replacement(occ_vrt_pairs, 2):
            params[index] = self.t2[i, j, a - nocc, b - nocc].real
            params[index + 1] = self.t2[i, j, a - nocc, b - nocc].imag
            index += 2
        # Final orbital rotation
        if self.final_orbital_rotation is not None:
            params[index:] = unitary_to_parameters(self.final_orbital_rotation)
        return params

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        if isinstance(nelec, int):
            return NotImplemented
        if copy:
            vec = vec.copy()

        linop = uccsd_restricted_linear_operator(
            self.t1, self.t2, norb=norb, nelec=nelec
        )
        vec = scipy.sparse.linalg.expm_multiply(linop, vec, traceA=0.0)

        if self.final_orbital_rotation is not None:
            vec = gates.apply_orbital_rotation(
                vec, self.final_orbital_rotation, norb=norb, nelec=nelec, copy=False
            )

        return vec

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        if isinstance(other, UCCSDOpRestricted):
            if not np.allclose(self.t1, other.t1, rtol=rtol, atol=atol):
                return False
            if not np.allclose(self.t2, other.t2, rtol=rtol, atol=atol):
                return False
            if (self.final_orbital_rotation is None) != (
                other.final_orbital_rotation is None
            ):
                return False
            if self.final_orbital_rotation is not None:
                return np.allclose(
                    cast(np.ndarray, self.final_orbital_rotation),
                    cast(np.ndarray, other.final_orbital_rotation),
                    rtol=rtol,
                    atol=atol,
                )
            return True
        return NotImplemented


@dataclass(frozen=True)
class UCCSDOpUnrestrictedReal(
    protocols.SupportsApplyUnitary, protocols.SupportsApproximateEquality
):
    """Real unrestricted unitary coupled cluster, singles and doubles operator."""

    t1: tuple[np.ndarray, np.ndarray]  # (t1_a, t1_b)
    # t1_a shape: (nocc_a, nvrt_a)
    # t1_b shape: (nocc_b, nvrt_b)
    t2: tuple[np.ndarray, np.ndarray, np.ndarray]  # (t1_aa, t1_ab, t1_bb)
    # t2_aa shape: (nocc_a, nocc_a, nvrt_a, nvrt_a)
    # t2_ab shape: (nocc_a, nocc_b, nvrt_a, nvrt_b)
    # t2_bb shape: (nocc_b, nocc_b, nvrt_b, nvrt_b)
    final_orbital_rotation: np.ndarray | None = None  # shape: (2, norb, norb)
    validate: InitVar[bool] = True
    rtol: InitVar[float] = 1e-5
    atol: InitVar[float] = 1e-8

    def __post_init__(self, validate: bool, rtol: float, atol: float):
        if any(np.iscomplexobj(t1) for t1 in self.t1):
            raise TypeError(
                "UCCSDOpRestricted only accepts real-valued t1 amplitudes. "
                "Please pass t1 amplitudes with a real-valued data type."
            )
        if any(np.iscomplexobj(t2) for t2 in self.t2):
            raise TypeError(
                "UCCSDOpRestricted only accepts real-valued t2 amplitudes. "
                "Please pass t2 amplitudes with a real-valued data type."
            )
        if validate:
            # Validate shapes
            t1_a, t1_b = self.t1
            nocc_a, nvrt_a = t1_a.shape
            nocc_b, nvrt_b = t1_b.shape
            norb_a = nocc_a + nvrt_a
            norb_b = nocc_b + nvrt_b
            if norb_a != norb_b:
                raise ValueError(
                    "t1_a and t1_b shapes not consistent. "
                    f"They sum to different numbers of orbitals, {norb_a} and {norb_b}."
                )
            norb = norb_a
            t2_aa, t2_ab, t2_bb = self.t2
            if t2_aa.shape != (nocc_a, nocc_a, nvrt_a, nvrt_a):
                raise ValueError(
                    "t2_aa shape not consistent with t1_a shape."
                    f"Expected {(nocc_a, nocc_a, nvrt_a, nvrt_a)} but got "
                    f"{t2_aa.shape}."
                )
            if t2_ab.shape != (nocc_a, nocc_b, nvrt_a, nvrt_b):
                raise ValueError(
                    "t2_ab shape not consistent with t1_a and t1_b shapes."
                    f"Expected {(nocc_a, nocc_b, nvrt_a, nvrt_b)} but got "
                    f"{t2_ab.shape}."
                )
            if t2_bb.shape != (nocc_b, nocc_b, nvrt_b, nvrt_b):
                raise ValueError(
                    "t2_bb shape not consistent with t1_b shape."
                    f"Expected {(nocc_b, nocc_b, nvrt_b, nvrt_b)} but got "
                    f"{t2_bb.shape}."
                )
            if self.final_orbital_rotation is not None:
                if self.final_orbital_rotation.shape != (2, norb, norb):
                    raise ValueError(
                        "Final orbital rotation shape not consistent with t1 shape. "
                        f"Expected {(2, norb, norb)} but got "
                        f"{self.final_orbital_rotation.shape}."
                    )
                orb_rot_a, orb_rot_b = self.final_orbital_rotation
                if not linalg.is_unitary(orb_rot_a, rtol=rtol, atol=atol):
                    raise ValueError("Final orbital rotation alpha was not unitary.")
                if not linalg.is_unitary(orb_rot_b, rtol=rtol, atol=atol):
                    raise ValueError("Final orbital rotation beta was not unitary.")

    @property
    def norb(self):
        """The number of spatial orbitals."""
        t1_a, _ = self.t1
        nocc_a, nvrt_a = t1_a.shape
        return nocc_a + nvrt_a

    @staticmethod
    def n_params(
        norb: int, nelec: tuple[int, int], *, with_final_orbital_rotation: bool = False
    ) -> int:
        """Return the number of parameters of an ansatz with given settings.

        Args:
            norb: The number of spatial orbitals.
            nelec: The numbers of spin alpha and spin beta fermions.
            with_final_orbital_rotation: Whether to include a final orbital rotation
                in the operator.

        Returns:
            The number of parameters of the ansatz.
        """
        nocc_a, nocc_b = nelec
        nvrt_a = norb - nocc_a
        nvrt_b = norb - nocc_b
        n_pairs_a = nocc_a * nvrt_a
        n_pairs_b = nocc_b * nvrt_b
        return (
            # t1_a and t1_b
            n_pairs_a
            + n_pairs_b
            # t2_aa, t2_ab, and t2_bb
            + n_pairs_a * (n_pairs_a + 1) // 2
            + n_pairs_a * n_pairs_b
            + n_pairs_b * (n_pairs_b + 1) // 2
            # final orbital rotation
            + with_final_orbital_rotation * 2 * norb**2
        )

    @staticmethod
    def from_parameters(
        params: np.ndarray,
        *,
        norb: int,
        nelec: tuple[int, int],
        with_final_orbital_rotation: bool = False,
    ) -> UCCSDOpUnrestrictedReal:
        r"""Initialize the UCCSD operator from a real-valued parameter vector.

        Args:
            params: The real-valued parameter vector.
            norb: The number of spatial orbitals.
            nelec: The numbers of spin alpha and spin beta fermions.
            with_final_orbital_rotation: Whether to include a final orbital rotation
                in the operator.

        Returns:
            The UCCSD operator constructed from the given parameters.

        Raises:
            ValueError: The number of parameters passed did not match the number
                expected based on the function inputs.
        """
        n_params = UCCSDOpUnrestrictedReal.n_params(
            norb, nelec, with_final_orbital_rotation=with_final_orbital_rotation
        )
        if len(params) != n_params:
            raise ValueError(
                "The number of parameters passed did not match the number expected "
                "based on the function inputs. "
                f"Expected {n_params} but got {len(params)}."
            )
        nocc_a, nocc_b = nelec
        nvrt_a = norb - nocc_a
        nvrt_b = norb - nocc_b
        t1_a = np.zeros((nocc_a, nvrt_a))
        t1_b = np.zeros((nocc_b, nvrt_b))
        t2_aa = np.zeros((nocc_a, nocc_a, nvrt_a, nvrt_a))
        t2_ab = np.zeros((nocc_a, nocc_b, nvrt_a, nvrt_b))
        t2_bb = np.zeros((nocc_b, nocc_b, nvrt_b, nvrt_b))
        occ_vrt_pairs_a = list(itertools.product(range(nocc_a), range(nocc_a, norb)))
        occ_vrt_pairs_b = list(itertools.product(range(nocc_b), range(nocc_b, norb)))
        index = 0
        # t1_a
        for i, a in occ_vrt_pairs_a:
            t1_a[i, a - nocc_a] = params[index]
            index += 1
        # t1_b
        for i, a in occ_vrt_pairs_b:
            t1_b[i, a - nocc_b] = params[index]
            index += 1
        # t2_aa
        for (i, a), (j, b) in itertools.combinations_with_replacement(
            occ_vrt_pairs_a, 2
        ):
            t2_aa[i, j, a - nocc_a, b - nocc_a] = params[index]
            t2_aa[j, i, b - nocc_a, a - nocc_a] = params[index]
            index += 1
        # t2_ab
        for (i, a), (j, b) in itertools.product(occ_vrt_pairs_a, occ_vrt_pairs_b):
            t2_ab[i, j, a - nocc_a, b - nocc_b] = params[index]
            index += 1
        # t2_bb
        for (i, a), (j, b) in itertools.combinations_with_replacement(
            occ_vrt_pairs_b, 2
        ):
            t2_bb[i, j, a - nocc_b, b - nocc_b] = params[index]
            t2_bb[j, i, b - nocc_b, a - nocc_b] = params[index]
            index += 1
        # Final orbital rotation
        final_orbital_rotation = None
        if with_final_orbital_rotation:
            n_params = norb**2
            orb_rot_a = unitary_from_parameters(params[index : index + n_params], norb)
            index += n_params
            orb_rot_b = unitary_from_parameters(params[index:], norb)
            final_orbital_rotation = np.stack((orb_rot_a, orb_rot_b))
        return UCCSDOpUnrestrictedReal(
            t1=(t1_a, t1_b),
            t2=(t2_aa, t2_ab, t2_bb),
            final_orbital_rotation=final_orbital_rotation,
        )

    def to_parameters(self) -> np.ndarray:
        r"""Convert the UCCSD operator to a real-valued parameter vector.

        Returns:
            The real-valued parameter vector.
        """
        t1_a, t1_b = self.t1
        t2_aa, t2_ab, t2_bb = self.t2
        nocc_a, nvrt_a = t1_a.shape
        nocc_b, _ = t1_b.shape
        norb = nocc_a + nvrt_a
        n_params = UCCSDOpUnrestrictedReal.n_params(
            norb,
            (nocc_a, nocc_b),
            with_final_orbital_rotation=self.final_orbital_rotation is not None,
        )
        params = np.zeros(n_params)
        occ_vrt_pairs_a = list(itertools.product(range(nocc_a), range(nocc_a, norb)))
        occ_vrt_pairs_b = list(itertools.product(range(nocc_b), range(nocc_b, norb)))
        index = 0
        # t1_a
        for i, a in occ_vrt_pairs_a:
            params[index] = t1_a[i, a - nocc_a]
            index += 1
        # t1_b
        for i, a in occ_vrt_pairs_b:
            params[index] = t1_b[i, a - nocc_b]
            index += 1
        # t2_aa
        for (i, a), (j, b) in itertools.combinations_with_replacement(
            occ_vrt_pairs_a, 2
        ):
            params[index] = t2_aa[i, j, a - nocc_a, b - nocc_a]
            index += 1
        # t2_ab
        for (i, a), (j, b) in itertools.product(occ_vrt_pairs_a, occ_vrt_pairs_b):
            params[index] = t2_ab[i, j, a - nocc_a, b - nocc_b]
            index += 1
        # t2_bb
        for (i, a), (j, b) in itertools.combinations_with_replacement(
            occ_vrt_pairs_b, 2
        ):
            params[index] = t2_bb[i, j, a - nocc_b, b - nocc_b]
            index += 1
        # Final orbital rotation
        if self.final_orbital_rotation is not None:
            orb_rot_a, orb_rot_b = self.final_orbital_rotation
            n_params = norb**2
            params[index : index + n_params] = unitary_to_parameters(orb_rot_a)
            index += n_params
            params[index:] = unitary_to_parameters(orb_rot_b)
        return params

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        if isinstance(nelec, int):
            return NotImplemented
        if copy:
            vec = vec.copy()

        linop = uccsd_unrestricted_linear_operator(
            self.t1, self.t2, norb=norb, nelec=nelec
        )
        vec = scipy.sparse.linalg.expm_multiply(linop, vec, traceA=0.0)

        if self.final_orbital_rotation is not None:
            vec = gates.apply_orbital_rotation(
                vec, self.final_orbital_rotation, norb=norb, nelec=nelec, copy=False
            )

        return vec

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        if isinstance(other, UCCSDOpUnrestrictedReal):
            if not all(
                np.allclose(t1_self, t1_other, rtol=rtol, atol=atol)
                for t1_self, t1_other in zip(self.t1, other.t1)
            ):
                return False
            if not all(
                np.allclose(t2_self, t2_other, rtol=rtol, atol=atol)
                for t2_self, t2_other in zip(self.t2, other.t2)
            ):
                return False
            if (self.final_orbital_rotation is None) != (
                other.final_orbital_rotation is None
            ):
                return False
            if self.final_orbital_rotation is not None:
                return np.allclose(
                    cast(np.ndarray, self.final_orbital_rotation),
                    cast(np.ndarray, other.final_orbital_rotation),
                    rtol=rtol,
                    atol=atol,
                )
            return True
        return NotImplemented
