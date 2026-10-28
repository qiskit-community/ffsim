# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Circuit building blocks shared by the Paldus and Schur transforms."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

from qiskit.circuit import CircuitInstruction, Qubit
from qiskit.circuit.library import MCXGate


def _controlled_increment(
    control: Qubit, register: Sequence[Qubit]
) -> Iterator[CircuitInstruction]:
    """Add 1 modulo 2**len(register) to a little-endian register if control is set."""
    for j in reversed(range(len(register))):
        yield CircuitInstruction(MCXGate(j + 1), (control, *register[:j], register[j]))


def _controlled_decrement(
    control: Qubit, register: Sequence[Qubit]
) -> Iterator[CircuitInstruction]:
    """Subtract 1 modulo 2**len(register) from a register if control is set."""
    yield from reversed(list(_controlled_increment(control, register)))


def _cg_angles(max_two_s: int) -> Iterator[tuple[int, int, float]]:
    r"""Yield the angles for coupling a spin 1/2 by a Y rotation.

    For incoming spin S and outgoing spin projection M, the rotation
    :math:`R_y(\theta)` with the yielded angle has
    :math:`\cos(\theta / 2) = c` and :math:`\sin(\theta / 2) = -s`, where
    :math:`c = \sqrt{(S + M + 1/2) / (2S + 1)}` and
    :math:`s = \sqrt{(S - M + 1/2) / (2S + 1)}` are Clebsch-Gordan coefficients.

    Yields a tuple (2S, 2M, angle) for each 2S up to max_two_s and each 2M with
    |2M| <= 2S + 1 and the opposite parity to 2S.
    """
    for two_s in range(max_two_s + 1):
        for two_m in range(-two_s - 1, two_s + 2, 2):
            angle = -2 * math.atan2(
                math.sqrt(two_s - two_m + 1), math.sqrt(two_s + two_m + 1)
            )
            yield two_s, two_m, angle
