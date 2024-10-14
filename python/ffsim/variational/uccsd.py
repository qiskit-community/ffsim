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
import scipy.linalg
import scipy.sparse.linalg

from ffsim import gates, hamiltonians, linalg, protocols
from ffsim.variational.util import (
    orbital_rotation_from_parameters,
    orbital_rotation_to_parameters,
)


@dataclass(frozen=True)
class UCCSDOpRestrictedReal:
    """Real-valued restricted unitary coupled cluster, singles and doubles operator."""

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
        if self.final_orbital_rotation is not None and np.iscomplexobj(
            self.final_orbital_rotation
        ):
            raise TypeError(
                "UCCSDOpRestricted only accepts a real-valued final "
                "orbital rotation. Please pass a final orbital rotation with a "
                "real-valued data type."
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
        # Final orbital rotation has norb * (norb - 1) // 2 parameters
        return (
            n_pairs
            + n_pairs * (n_pairs + 1) // 2
            + with_final_orbital_rotation * norb * (norb - 1) // 2
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
            final_orbital_rotation = orbital_rotation_from_parameters(
                params[index:], norb, real=True
            )
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
            params[index:] = orbital_rotation_to_parameters(
                self.final_orbital_rotation, real=True
            )
        return params

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        if isinstance(nelec, int):
            return NotImplemented
        if copy:
            vec = vec.copy()

        nocc, _ = self.t1.shape
        assert nelec == (nocc, nocc)

        one_body_tensor = np.zeros((norb, norb))
        two_body_tensor = np.zeros((norb, norb, norb, norb))
        one_body_tensor[:nocc, nocc:] = self.t1
        one_body_tensor[nocc:, :nocc] = -self.t1.T
        two_body_tensor[nocc:, :nocc, nocc:, :nocc] = self.t2.transpose(2, 0, 3, 1)
        two_body_tensor[:nocc, nocc:, :nocc, nocc:] = -self.t2.transpose(0, 2, 1, 3)

        linop = protocols.linear_operator(
            hamiltonians.MolecularHamiltonian(
                one_body_tensor=one_body_tensor, two_body_tensor=two_body_tensor
            ),
            norb=norb,
            nelec=nelec,
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
