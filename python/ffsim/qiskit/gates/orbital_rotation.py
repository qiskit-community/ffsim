# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Orbital rotation gate."""

from __future__ import annotations

import cmath
import math
from collections.abc import Iterator, Sequence

import numpy as np
from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.library import PhaseGate, XXPlusYYGate

from ffsim import linalg


def _validate_orbital_rotation(
    mat: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
    rtol: float,
    atol: float,
) -> None:
    if isinstance(mat, np.ndarray) and mat.ndim == 2:
        if not linalg.is_unitary(mat, rtol=rtol, atol=atol):
            raise ValueError("The input orbital rotation matrix was not unitary.")
    else:
        mat_a, mat_b = mat
        if mat_a is not None and not linalg.is_unitary(mat_a, rtol=rtol, atol=atol):
            raise ValueError(
                "The input orbital rotation matrix for spin alpha was not unitary."
            )
        if mat_b is not None and not linalg.is_unitary(mat_b, rtol=rtol, atol=atol):
            raise ValueError(
                "The input orbital rotation matrix for spin beta was not unitary."
            )


class OrbitalRotationJW(Gate):
    r"""Orbital rotation under the Jordan-Wigner transformation.

    An orbital rotation maps creation operators as

    .. math::

        a^\dagger_{i\sigma} \mapsto \sum_{j} U_{ji} a^\dagger_{j\sigma}

    where :math:`U` is a unitary matrix. This is equivalent to applying the
    transformation given by

    .. math::

        \prod_{\sigma}
        \exp\left(\sum_{ij} \log(U)_{ij} a^\dagger_{i\sigma} a_{j\sigma}\right)

    This gate assumes that qubits are ordered such that the first ``norb`` qubits
    correspond to the alpha orbitals and the last ``norb`` qubits correspond to the
    beta orbitals.
    """

    def __init__(
        self,
        norb: int,
        orbital_rotation: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
        *,
        tol: float = 1e-12,
        max_givens: int | None = None,
        max_layers: int | None = None,
        label: str | None = None,
        validate: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        **optimize_kwargs,
    ):
        """Create new orbital rotation gate.

        Args:
            norb: The number of spatial orbitals.
            orbital_rotation: The orbital rotation.
                You can pass either a single Numpy array specifying the orbital rotation
                to apply to both spin sectors, or you can pass a pair of Numpy arrays
                specifying independent orbital rotations for spin alpha and spin beta.
                If passing a pair, you can use ``None`` for one of the
                values in the pair to indicate that no operation should be applied to
                that spin sector.
            tol: Tolerance for the Givens decomposition of the orbital rotation.
                Matrix entries smaller than this value will be treated as equal to zero.
            max_givens: The maximum number of Givens rotations to use. If specified, the
                decomposition is compressed to use at most this many Givens rotations,
                and the resulting gate only approximates the orbital rotation. See
                :func:`~ffsim.linalg.givens_decomposition` for details.
            max_layers: The maximum number of brickwork layers to use. If specified, the
                decomposition is compressed to use at most this many layers, and the
                resulting gate only approximates the orbital rotation. See
                :func:`~ffsim.linalg.givens_decomposition` for details.
            label: The label of the gate.
            validate: Whether to check that the input orbital rotation(s) is unitary
                and raise an error if it isn't.
            rtol: Relative numerical tolerance for input validation.
            atol: Absolute numerical tolerance for input validation.
            optimize_kwargs: Keyword arguments to pass to
                :func:`scipy.optimize.minimize`, which performs the optimization when
                the decomposition is compressed.

        Raises:
            ValueError: The input matrix is not unitary.
        """
        if validate:
            _validate_orbital_rotation(orbital_rotation, rtol=rtol, atol=atol)
        self.norb = norb
        if isinstance(orbital_rotation, np.ndarray) and orbital_rotation.ndim == 2:
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
        self.tol = tol
        self.max_givens = max_givens
        self.max_layers = max_layers
        self.optimize_kwargs = optimize_kwargs
        super().__init__("orb_rot_jw", 2 * norb, [], label=label)

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        norb = len(qubits) // 2
        alpha_qubits = qubits[:norb]
        beta_qubits = qubits[norb:]
        for instruction in _orbital_rotation_jw(
            alpha_qubits,
            self.orbital_rotation_a,
            tol=self.tol,
            max_givens=self.max_givens,
            max_layers=self.max_layers,
            **self.optimize_kwargs,
        ):
            circuit.append(instruction)
        for instruction in _orbital_rotation_jw(
            beta_qubits,
            self.orbital_rotation_b,
            tol=self.tol,
            max_givens=self.max_givens,
            max_layers=self.max_layers,
            **self.optimize_kwargs,
        ):
            circuit.append(instruction)
        self.definition = circuit

    def inverse(self):
        """Inverse gate."""
        return OrbitalRotationJW(
            self.norb,
            (self.orbital_rotation_a.T.conj(), self.orbital_rotation_b.T.conj()),
        )


class OrbitalRotationSpinlessJW(Gate):
    r"""Orbital rotation under the Jordan-Wigner transformation, spinless version.

    Like :class:`OrbitalRotationJW` but only acts on a single spin species.
    """

    def __init__(
        self,
        norb: int,
        orbital_rotation: np.ndarray,
        *,
        tol: float = 1e-12,
        max_givens: int | None = None,
        max_layers: int | None = None,
        label: str | None = None,
        validate: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        **optimize_kwargs,
    ):
        """Create new orbital rotation gate.

        Args:
            norb: The number of spatial orbitals.
            orbital_rotation: The orbital rotation.
            tol: Tolerance for the Givens decomposition of the orbital rotation.
                Matrix entries smaller than this value will be treated as equal to zero.
            max_givens: The maximum number of Givens rotations to use. If specified, the
                decomposition is compressed to use at most this many Givens rotations,
                and the resulting gate only approximates the orbital rotation. See
                :func:`~ffsim.linalg.givens_decomposition` for details.
            max_layers: The maximum number of brickwork layers to use. If specified, the
                decomposition is compressed to use at most this many layers, and the
                resulting gate only approximates the orbital rotation. See
                :func:`~ffsim.linalg.givens_decomposition` for details.
            label: The label of the gate.
            validate: Whether to check that the input orbital rotation(s) is unitary
                and raise an error if it isn't.
            rtol: Relative numerical tolerance for input validation.
            atol: Absolute numerical tolerance for input validation.
            optimize_kwargs: Keyword arguments to pass to
                :func:`scipy.optimize.minimize`, which performs the optimization when
                the decomposition is compressed.

        Raises:
            ValueError: The input matrix is not unitary.
        """
        if validate and orbital_rotation is not None:
            if not linalg.is_unitary(orbital_rotation, rtol=rtol, atol=atol):
                raise ValueError("The input orbital rotation matrix was not unitary.")

        self.norb = norb
        self.orbital_rotation = orbital_rotation
        self.tol = tol
        self.max_givens = max_givens
        self.max_layers = max_layers
        self.optimize_kwargs = optimize_kwargs
        super().__init__("orb_rot_spinless_jw", norb, [], label=label)

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        for instruction in _orbital_rotation_jw(
            qubits,
            self.orbital_rotation,
            tol=self.tol,
            max_givens=self.max_givens,
            max_layers=self.max_layers,
            **self.optimize_kwargs,
        ):
            circuit.append(instruction)
        self.definition = circuit

    def inverse(self):
        """Inverse gate."""
        return OrbitalRotationSpinlessJW(self.norb, self.orbital_rotation.T.conj())


def _orbital_rotation_jw(
    qubits: Sequence[Qubit],
    orbital_rotation: np.ndarray,
    tol: float,
    max_givens: int | None = None,
    max_layers: int | None = None,
    **optimize_kwargs,
) -> Iterator[CircuitInstruction]:
    givens_rotations, phase_shifts = linalg.givens_decomposition(
        orbital_rotation,
        tol=tol,
        max_givens=max_givens,
        max_layers=max_layers,
        **optimize_kwargs,
    )
    for c, s, i, j in givens_rotations:
        r, phi = cmath.polar(s)
        theta = math.atan2(r, c)
        if theta:
            yield CircuitInstruction(
                XXPlusYYGate(2 * theta, phi - 0.5 * math.pi),
                (qubits[i], qubits[j]),
            )
    for i, phase_shift in enumerate(phase_shifts):
        phase = cmath.phase(phase_shift)
        if phase:
            yield CircuitInstruction(PhaseGate(phase), (qubits[i],))
