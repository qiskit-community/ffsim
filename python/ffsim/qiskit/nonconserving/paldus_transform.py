# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Paldus transform."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.library import CXGate, CZGate, RCCXGate, UCRYGate, XGate

from ffsim.qiskit.nonconserving._util import _cg_angles, _controlled_increment


class PaldusTransformJW(Gate):
    r"""Quantum Paldus transform under the Jordan-Wigner transformation.

    Maps Gelfand-Tsetlin (GT) states, also known as Yamanouchi-Kotani configuration
    state functions, to the unitary group approach (UGA) basis, in which the particle
    number :math:`N`, total spin :math:`S`, spin projection :math:`M`, and step vector
    :math:`\mathbf{d}` are each stored in a separate register:

    .. math::
        U_P \left( |0\rangle_N |0\rangle_S |0\rangle_M
            \otimes |N, S, M; \mathbf{d}\rangle \right)
        = |N\rangle_N |2S\rangle_S |2M\rangle_M |\mathbf{d}\rangle.

    The GT state :math:`|N, S, M; \mathbf{d}\rangle` is obtained by coupling the
    spatial orbitals one at a time, in order, using Clebsch-Gordan coefficients
    with the Condon-Shortley phase convention. Orbital :math:`i` contributes
    :math:`1`, :math:`a^\dagger_{i\sigma}`, or
    :math:`a^\dagger_{i\alpha} a^\dagger_{i\beta}` according to whether it is empty,
    singly occupied, or doubly occupied, and the creation operators are ordered by
    orbital index. The circuit is a sequence of Clebsch-Gordan transforms, one per
    spatial orbital, as described in `arXiv:2506.09151`_.

    The gate acts on ``2 * norb + num_qubits_n + num_qubits_s + num_qubits_m``
    qubits, laid out as follows:

    - The first ``2 * norb`` qubits hold the fermionic modes, with the first ``norb``
      qubits for spin alpha and the next ``norb`` qubits for spin beta. On output,
      they hold the step vector. The step for orbital :math:`i` is stored on qubits
      :math:`i` (alpha) and :math:`\text{norb} + i` (beta) as 00 for an empty orbital,
      10 for a step that raises the spin by 1/2, 01 for a step that lowers the spin
      by 1/2, and 11 for a doubly occupied orbital.
    - The next ``num_qubits_n`` qubits hold :math:`N`.
    - The next ``num_qubits_s`` qubits hold :math:`2S`.
    - The last ``num_qubits_m`` qubits hold :math:`2M` in two's complement.

    Each of the :math:`N`, :math:`S`, and :math:`M` registers stores its least
    significant bit on its first qubit. The action of the gate is only specified
    when these registers start in the all-zeros state.

    .. _arXiv:2506.09151: https://arxiv.org/abs/2506.09151
    """

    def __init__(self, norb: int, *, label: str | None = None):
        """Create new Paldus transform gate.

        Args:
            norb: The number of spatial orbitals.
            label: Optional label for the gate.

        Raises:
            ValueError: norb must be positive.
        """
        if norb < 1:
            raise ValueError(f"norb must be positive. Got {norb}.")
        self.norb = norb
        self.num_qubits_n = (2 * norb).bit_length()
        self.num_qubits_s = norb.bit_length()
        self.num_qubits_m = (2 * norb).bit_length()
        super().__init__(
            "paldus_transform_jw",
            2 * norb + self.num_qubits_n + self.num_qubits_s + self.num_qubits_m,
            [],
            label=label,
        )

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits, name="q")
        circuit = QuantumCircuit(qubits, name=self.name)
        norb = self.norb
        alpha_qubits = qubits[:norb]
        beta_qubits = qubits[norb : 2 * norb]
        start = 2 * norb
        n_qubits = qubits[start : start + self.num_qubits_n]
        start += self.num_qubits_n
        s_qubits = qubits[start : start + self.num_qubits_s]
        start += self.num_qubits_s
        m_qubits = qubits[start : start + self.num_qubits_m]

        # Reorder creation operators from all alpha before all beta to orbital order
        for j in range(norb):
            for i in range(j):
                circuit.append(CZGate(), (beta_qubits[i], alpha_qubits[j]))

        # The low bits of N, 2S, and 2M are all equal to the parity of the number of
        # singly occupied orbitals. The low bit of N is copied from that of 2S at the
        # end, so only N / 2, rounded down, is tracked.
        s_low = s_qubits[0]
        for i, (a, b) in enumerate(zip(alpha_qubits, beta_qubits)):
            # Before this step, 2S <= i and |2M| <= i. After it, 2S <= i + 1 and
            # |2M| <= i + 1, so only the low bits of the registers are needed. The
            # M register grows by at most one bit, so sign extend it.
            n_s = i.bit_length()
            m_bits = m_qubits[: (i + 1).bit_length() + 1]
            if i and (i + 1).bit_length() > i.bit_length():
                circuit.append(CXGate(), (m_bits[-2], m_bits[-1]))
            # Set alpha to the parity of the orbital occupation. Then alpha is 1 for
            # inputs 01 and 10, and beta distinguishes them.
            circuit.append(CXGate(), (b, a))
            # Update M to its outgoing value
            for instruction in _controlled_add_sign(a, b, m_bits):
                circuit.append(instruction)
            # Rotate by Clebsch-Gordan coefficients
            for instruction in _cg_rotation(a, b, s_qubits[:n_s], m_bits, max_two_s=i):
                circuit.append(instruction)
            # Update N / 2, rounded down, to its outgoing value. It increases by 1 if
            # the orbital is doubly occupied, or if it is singly occupied and the low
            # bit of N is 1. Compute this condition in the low bit of 2S, which is the
            # incoming low bit of N. The relative phases of the RCCX gate cancel when
            # it is undone, because the increment in between only uses its qubits as
            # controls.
            n_bits = n_qubits[1 : (i + 1).bit_length() + 1]
            carry = [
                CircuitInstruction(CXGate(), (s_low, b)),
                CircuitInstruction(XGate(), (a,)),
                CircuitInstruction(RCCXGate(), (a, b, s_low)),
                CircuitInstruction(XGate(), (a,)),
            ]
            for instruction in carry:
                circuit.append(instruction)
            for instruction in _controlled_increment(s_low, n_bits):
                circuit.append(instruction)
            for instruction in reversed(carry):
                circuit.append(instruction)
            # Update S to its outgoing value, which is at most i + 1
            for instruction in _controlled_add_sign(
                a, b, s_qubits[: (i + 1).bit_length()]
            ):
                circuit.append(instruction)
            circuit.append(CXGate(), (b, a))
        circuit.append(CXGate(), (s_low, n_qubits[0]))

        self.definition = circuit


def _controlled_add_sign(
    control: Qubit, sign: Qubit, register: Sequence[Qubit]
) -> Iterator[CircuitInstruction]:
    """Add 1 to a register if sign is 0, or subtract 1 if sign is 1, if control is set.

    The register is little-endian and the addition is modulo 2**len(register).
    """
    # Subtracting 1 is the same as adding 1 to the bitwise complement and taking the
    # bitwise complement of the result
    for q in register:
        yield CircuitInstruction(CXGate(), (sign, q))
    yield from _controlled_increment(control, register)
    for q in register:
        yield CircuitInstruction(CXGate(), (sign, q))


def _cg_rotation(
    parity: Qubit,
    beta: Qubit,
    s_qubits: Sequence[Qubit],
    m_qubits: Sequence[Qubit],
    max_two_s: int,
) -> Iterator[CircuitInstruction]:
    r"""Givens rotation on an orbital multiplexed on the S and M registers.

    Here parity holds the XOR of the alpha and beta occupations of the orbital. For
    incoming spin S and outgoing spin projection M, the rotation maps

    .. math::
        |01\rangle \mapsto c |01\rangle + s |10\rangle, \quad
        |10\rangle \mapsto -s |01\rangle + c |10\rangle,

    where :math:`c = \sqrt{(S + M + 1/2) / (2S + 1)}` and
    :math:`s = \sqrt{(S - M + 1/2) / (2S + 1)}` are Clebsch-Gordan coefficients.
    Here the bit strings list the alpha qubit first. The registers hold 2S and 2M
    (in two's complement) and may be truncated to their low bits, as long as they
    can represent all 2S up to max_two_s and all 2M with |2M| <= max_two_s + 1.
    """
    # The rotation is only nontrivial if parity is 1, and then the low bit of 2M is
    # the complement of the low bit of 2S. So the low bit of 2M is not a control.
    # Parity is the least significant control.
    n_s = len(s_qubits)
    n_m = len(m_qubits)
    angles = [0.0] * (1 << (n_s + n_m))
    for two_s, two_m, angle in _cg_angles(max_two_s):
        angles[1 | (two_s << 1) | ((two_m % (1 << n_m)) >> 1 << (n_s + 1))] = angle
    yield CircuitInstruction(UCRYGate(angles), (beta, parity, *s_qubits, *m_qubits[1:]))
