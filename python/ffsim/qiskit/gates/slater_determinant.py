# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Slater determinant and Hartree-Fock preparation gates."""

from __future__ import annotations

import cmath
import math
from collections.abc import Iterator, Sequence

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import CircuitInstruction, Gate, Qubit
from qiskit.circuit.library import XGate, XXPlusYYGate
from scipy.linalg.lapack import zrot

from ffsim import linalg
from ffsim.linalg.givens import GivensRotation, zrotg
from ffsim.qiskit.gates.orbital_rotation import _validate_orbital_rotation


class PrepareHartreeFockJW(Gate):
    r"""Gate that prepares the Hartree-Fock state (under JWT) from the all zeros state.

    This gate assumes the Jordan-Wigner transformation (JWT).

    This gate is meant to be applied to the all zeros state. It decomposes simply as
    a sequence of X gates that prepares the Hartree-Fock electronic configuration.

    This gate assumes that qubits are ordered such that the first `norb` qubits
    correspond to the alpha orbitals and the last `norb` qubits correspond to the
    beta orbitals.
    """

    def __init__(
        self, norb: int, nelec: tuple[int, int], label: str | None = None
    ) -> None:
        """Create new Hartree-Fock state preparation gate.

        Args:
            norb: The number of spatial orbitals.
            nelec: The number of alpha and beta electrons.
            label: The label of the gate.
        """
        self.norb = norb
        self.nelec = nelec
        super().__init__("hartree_fock_jw", 2 * norb, [], label=label)

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        n_alpha, n_beta = self.nelec
        circuit.append(
            PrepareSlaterDeterminantJW(self.norb, (range(n_alpha), range(n_beta))),
            qubits,
        )
        self.definition = circuit


class PrepareHartreeFockSpinlessJW(Gate):
    r"""Prepare the Hartree-Fock state (under JWT) from the zero state, spinless.

    Like :class:`PrepareHartreeFockJW` but only acts on a single spin species.
    """

    def __init__(self, norb: int, nelec: int, label: str | None = None) -> None:
        """Create new Hartree-Fock state preparation gate.

        Args:
            norb: The number of spatial orbitals.
            nelec: The number of electrons.
            label: The label of the gate.
        """
        self.norb = norb
        self.nelec = nelec
        super().__init__("hartree_fock_spinless_jw", norb, [], label=label)

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        circuit.append(
            PrepareSlaterDeterminantSpinlessJW(self.norb, range(self.nelec)),
            qubits,
        )
        self.definition = circuit


class PrepareSlaterDeterminantJW(Gate):
    r"""Gate that prepares a Slater determinant (under JWT) from the all zeros state.

    This gate assumes the Jordan-Wigner transformation (JWT).

    A Slater determinant is a state of the form

    .. math::

        \mathcal{U} \lvert x \rangle,

    where :math:`\mathcal{U}` is an
    :doc:`orbital rotation </explanations/orbital-rotation>` and
    :math:`\lvert x \rangle` is an electronic configuration (computational basis state).
    The reason this gate exists (when :class:`OrbitalRotationJW` already exists) is that
    the preparation of a Slater determinant has a more optimized circuit than a generic
    orbital rotation.

    This gate is meant to be applied to the all zeros state. Its behavior when applied
    to any other state is not guaranteed. The global phase of the prepared state may
    be arbitrary.

    This gate assumes that qubits are ordered such that the first `norb` qubits
    correspond to the alpha orbitals and the last `norb` qubits correspond to the
    beta orbitals.

    Reference: `arXiv:1711.05395`_

    .. _arXiv:1711.05395: https://arxiv.org/abs/1711.05395
    """

    def __init__(
        self,
        norb: int,
        occupied_orbitals: tuple[Sequence[int], Sequence[int]],
        orbital_rotation: np.ndarray
        | tuple[np.ndarray | None, np.ndarray | None]
        | None = None,
        *,
        label: str | None = None,
        validate: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> None:
        """Create new Slater determinant preparation gate.

        Args:
            norb: The number of spatial orbitals.
            occupied_orbitals: The occupied orbitals in the electonic configuration.
                This is a pair of lists of integers, where the first list specifies the
                spin alpha orbitals and the second list specifies the spin beta
                orbitals.
            orbital_rotation: The optional orbital rotation.
                You can pass either a single Numpy array specifying the orbital rotation
                to apply to both spin sectors, or you can pass a pair of Numpy arrays
                specifying independent orbital rotations for spin alpha and spin beta.
                If passing a pair, you can use ``None`` for one of the
                values in the pair to indicate that no operation should be applied to
                that spin sector.
            label: The label of the gate.
            validate: Whether to check that the input orbital rotation(s) is unitary
                and raise an error if it isn't.
            rtol: Relative numerical tolerance for input validation.
            atol: Absolute numerical tolerance for input validation.

        Raises:
            ValueError: The input orbital rotation matrix is not unitary.
        """
        if validate and orbital_rotation is not None:
            _validate_orbital_rotation(orbital_rotation, rtol=rtol, atol=atol)
        self.norb = norb
        self.occupied_orbitals = occupied_orbitals
        if orbital_rotation is None:
            self.orbital_rotation_a = np.eye(norb)
            self.orbital_rotation_b = np.eye(norb)
        elif isinstance(orbital_rotation, np.ndarray):
            self.orbital_rotation_a = orbital_rotation
            self.orbital_rotation_b = orbital_rotation
        else:
            orbital_rotation_a, orbital_rotation_b = orbital_rotation
            if orbital_rotation_a is None:
                self.orbital_rotation_a = np.eye(self.norb)
            else:
                self.orbital_rotation_a = orbital_rotation_a
            if orbital_rotation_b is None:
                self.orbital_rotation_b = np.eye(self.norb)
            else:
                self.orbital_rotation_b = orbital_rotation_b
        super().__init__("slater_jw", 2 * norb, [], label=label)

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        alpha_qubits = qubits[: self.norb]
        beta_qubits = qubits[self.norb :]
        occ_a, occ_b = self.occupied_orbitals

        if np.array_equal(self.orbital_rotation_a, np.eye(self.norb)):
            for instruction in _prepare_configuration_jw(alpha_qubits, occ_a):
                circuit.append(instruction)
        else:
            for instruction in _prepare_slater_determinant_jw(
                alpha_qubits, self.orbital_rotation_a.T[list(occ_a)]
            ):
                circuit.append(instruction)

        if np.array_equal(self.orbital_rotation_b, np.eye(self.norb)):
            for instruction in _prepare_configuration_jw(beta_qubits, occ_b):
                circuit.append(instruction)
        else:
            for instruction in _prepare_slater_determinant_jw(
                beta_qubits, self.orbital_rotation_b.T[list(occ_b)]
            ):
                circuit.append(instruction)
        self.definition = circuit


class PrepareSlaterDeterminantSpinlessJW(Gate):
    r"""Prepare a Slater determinant (under JWT) from the zero state, spinless version.

    Like :class:`PrepareSlaterDeterminantJW` but only acts on a single spin species.
    """

    def __init__(
        self,
        norb: int,
        occupied_orbitals: Sequence[int],
        orbital_rotation: np.ndarray | None = None,
        *,
        label: str | None = None,
        validate: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> None:
        """Create new Slater determinant preparation gate.

        Args:
            norb: The number of spatial orbitals.
            occupied_orbitals: The occupied orbitals in the electonic configuration.
            orbital_rotation: The optional orbital rotation.
            label: The label of the gate.
            validate: Whether to check that the input orbital rotation(s) is unitary
                and raise an error if it isn't.
            rtol: Relative numerical tolerance for input validation.
            atol: Absolute numerical tolerance for input validation.

        Raises:
            ValueError: The input orbital rotation matrix is not unitary.
        """
        if validate and orbital_rotation is not None:
            if not linalg.is_unitary(orbital_rotation, rtol=rtol, atol=atol):
                raise ValueError("The input orbital rotation matrix was not unitary.")

        self.norb = norb
        self.occupied_orbitals = occupied_orbitals
        if orbital_rotation is None:
            self.orbital_rotation = np.eye(norb)
        else:
            self.orbital_rotation = orbital_rotation
        super().__init__("slater_spinless_jw", norb, [], label=label)

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)

        if np.array_equal(self.orbital_rotation, np.eye(self.norb)):
            for instruction in _prepare_configuration_jw(
                qubits, self.occupied_orbitals
            ):
                circuit.append(instruction)
        else:
            for instruction in _prepare_slater_determinant_jw(
                qubits, self.orbital_rotation.T[list(self.occupied_orbitals)]
            ):
                circuit.append(instruction)

        self.definition = circuit


def _prepare_configuration_jw(
    qubits: Sequence[Qubit], occupied_orbitals: Sequence[int]
) -> Iterator[CircuitInstruction]:
    for orb in occupied_orbitals:
        yield CircuitInstruction(XGate(), (qubits[orb],))


def _prepare_slater_determinant_jw(
    qubits: Sequence[Qubit], orbital_coeffs: np.ndarray
) -> Iterator[CircuitInstruction]:
    m, n = orbital_coeffs.shape

    # set the first n_particles qubits to 1
    yield from _prepare_configuration_jw(qubits, range(m))

    # if all orbitals are filled, no further operations are needed
    if m == n:
        return

    # yield Givens rotations
    givens_rotations = _givens_decomposition_slater(orbital_coeffs)
    for c, s, i, j in givens_rotations:
        yield CircuitInstruction(
            XXPlusYYGate(2 * math.acos(c), cmath.phase(s) - 0.5 * math.pi),
            (qubits[i], qubits[j]),
        )


def _givens_decomposition_slater(orbital_coeffs: np.ndarray) -> list[GivensRotation]:
    m, n = orbital_coeffs.shape

    current_matrix = orbital_coeffs.astype(complex, copy=True)

    # zero out top right corner by rotating rows; this is a no-op
    for j in reversed(range(n - m + 1, n)):
        # Zero out entries in column j
        for i in range(m - n + j):
            # Zero out entry in row i if needed
            if not cmath.isclose(current_matrix[i, j], 0):
                c, s = zrotg(current_matrix[i + 1, j], current_matrix[i, j])
                (
                    current_matrix[i + 1],
                    current_matrix[i],
                ) = zrot(
                    current_matrix[i + 1],
                    current_matrix[i],
                    c,
                    s,
                )

    # decompose matrix into Givens rotations
    rotations = []
    for i in range(m):
        # zero out the columns in row i
        for j in range(n - m + i, i, -1):
            if not cmath.isclose(current_matrix[i, j], 0):
                # zero out element j of row i
                c, s = zrotg(current_matrix[i, j - 1], current_matrix[i, j])
                rotations.append(GivensRotation(c, s, j, j - 1))
                (
                    current_matrix[:, j - 1],
                    current_matrix[:, j],
                ) = zrot(
                    current_matrix[:, j - 1],
                    current_matrix[:, j],
                    c,
                    s,
                )

    return rotations[::-1]
