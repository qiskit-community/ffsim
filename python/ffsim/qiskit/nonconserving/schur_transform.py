# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Schur transform."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.library import CXGate, UCRYGate, XGate

from ffsim.qiskit.nonconserving._util import (
    _cg_angles,
    _controlled_decrement,
    _controlled_increment,
)


class SchurTransform(Gate):
    r"""Quantum Schur transform on spin-1/2 qubits.

    Maps the Schur states of :math:`n` spin-1/2 particles, also known as spin
    eigenfunctions, to a basis in which the total spin :math:`S`, spin projection
    :math:`M`, and Yamanouchi symbol :math:`Y` are each stored in a separate
    register:

    .. math::
        U_\text{Sch} \left( |S, M; Y\rangle \otimes |0\rangle_S |0\rangle_M \right)
        = |Y\rangle |2S\rangle_S |2M\rangle_M.

    The Schur state :math:`|S, M; Y\rangle` is obtained by coupling the spins one at
    a time, in order, using Clebsch-Gordan coefficients with the Condon-Shortley
    phase convention. Bit :math:`i` of the Yamanouchi symbol is 0 if coupling spin
    :math:`i` raises the total spin by 1/2, and 1 if it lowers the total spin by
    1/2. The circuit is a sequence of Clebsch-Gordan transforms, one per spin, as in
    `Bacon, Chuang, and Harrow`_. It is the special case of
    :class:`PaldusTransformJW` in which every orbital is singly occupied, as
    described in `arXiv:2506.09151`_.

    The gate acts on ``num_spins + num_qubits_s + num_qubits_m`` qubits, laid out
    as follows:

    - The first ``num_spins`` qubits hold the spins, with :math:`|0\rangle` for spin
      up and :math:`|1\rangle` for spin down. On output, they hold the Yamanouchi
      symbol.
    - The next ``num_qubits_s`` qubits hold :math:`2S`.
    - The last ``num_qubits_m`` qubits hold :math:`2M` in two's complement.

    Each of the :math:`S` and :math:`M` registers stores its least significant bit
    on its first qubit. The action of the gate is only specified when these
    registers start in the all-zeros state.

    .. _Bacon, Chuang, and Harrow: https://doi.org/10.1103/PhysRevLett.97.170502
    .. _arXiv:2506.09151: https://arxiv.org/abs/2506.09151
    """

    def __init__(self, num_spins: int, *, label: str | None = None):
        """Create new Schur transform gate.

        Args:
            num_spins: The number of spin-1/2 particles.
            label: Optional label for the gate.

        Raises:
            ValueError: num_spins must be positive.
        """
        if num_spins < 1:
            raise ValueError(f"num_spins must be positive. Got {num_spins}.")
        self.num_spins = num_spins
        self.num_qubits_s = num_spins.bit_length()
        self.num_qubits_m = (2 * num_spins).bit_length()
        super().__init__(
            "schur_transform",
            num_spins + self.num_qubits_s + self.num_qubits_m,
            [],
            label=label,
        )

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits, name="q")
        circuit = QuantumCircuit(qubits, name=self.name)
        spin_qubits = qubits[: self.num_spins]
        s_qubits = qubits[self.num_spins : self.num_spins + self.num_qubits_s]
        m_qubits = qubits[self.num_spins + self.num_qubits_s :]

        # Before coupling spin i, the low bits of 2S and 2M are both i % 2. So only
        # 2S / 2 and 2M / 2, rounded down, are tracked, and the low bits are set at
        # the end.
        for i, q in enumerate(spin_qubits):
            # Before this step, 2S <= i and |2M| <= i. After it, 2S <= i + 1 and
            # |2M| <= i + 1, so only the low bits of the registers are needed. The
            # M register grows by at most one bit, so sign extend it.
            n_s = i.bit_length()
            s_bits = s_qubits[1 : (i + 1).bit_length()]
            m_bits = m_qubits[1 : (i + 1).bit_length() + 1]
            if i and (i + 1).bit_length() > i.bit_length():
                circuit.append(CXGate(), (m_bits[-2], m_bits[-1]))
            # Update M to its outgoing value, adding 1 to 2M for spin up and
            # subtracting 1 for spin down
            for instruction in _update_high_bits(q, m_bits, odd=i % 2 == 1):
                circuit.append(instruction)
            # Rotate by Clebsch-Gordan coefficients. The low bits of 2S and 2M are
            # known, so they are not controls.
            s_controls = s_qubits[1:n_s]
            angles = [0.0] * (1 << (len(s_controls) + len(m_bits)))
            for two_s, two_m, angle in _cg_angles(max_two_s=i):
                if two_s % 2 == i % 2:
                    index = (two_s >> 1) | (
                        (two_m % (1 << (len(m_bits) + 1))) >> 1 << len(s_controls)
                    )
                    angles[index] = angle
            circuit.append(
                CircuitInstruction(UCRYGate(angles), (q, *s_controls, *m_bits))
            )
            # Update S to its outgoing value, which is at most i + 1, adding 1 to 2S
            # for a Yamanouchi bit of 0 and subtracting 1 for a Yamanouchi bit of 1
            for instruction in _update_high_bits(q, s_bits, odd=i % 2 == 1):
                circuit.append(instruction)
        if self.num_spins % 2:
            circuit.append(XGate(), (s_qubits[0],))
            circuit.append(XGate(), (m_qubits[0],))

        self.definition = circuit


def _update_high_bits(
    qubit: Qubit, register: Sequence[Qubit], odd: bool
) -> Iterator[CircuitInstruction]:
    """Add 1 to a value if qubit is 0, or subtract 1 from it if qubit is 1.

    The register holds the value divided by 2, rounded down, and odd is whether the
    value is odd before the update. If the value is odd, adding 1 increments the
    register and subtracting 1 leaves it unchanged. If the value is even, adding 1
    leaves the register unchanged and subtracting 1 decrements it.
    """
    if odd:
        yield CircuitInstruction(XGate(), (qubit,))
        yield from _controlled_increment(qubit, register)
        yield CircuitInstruction(XGate(), (qubit,))
    else:
        yield from _controlled_decrement(qubit, register)
