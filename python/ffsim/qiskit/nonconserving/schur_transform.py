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

from qiskit.circuit import CircuitInstruction, Gate, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UCRYGate, XGate

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

        for i, q in enumerate(spin_qubits):
            # Update M to its outgoing value, adding 1 for spin up and subtracting 1
            # for spin down. All bits are updated to keep the two's complement sign
            # extension correct.
            circuit.append(XGate(), (q,))
            for instruction in _controlled_increment(q, m_qubits):
                circuit.append(instruction)
            circuit.append(XGate(), (q,))
            for instruction in _controlled_decrement(q, m_qubits):
                circuit.append(instruction)
            # Rotate by Clebsch-Gordan coefficients. Before this step, 2S <= i and
            # |2M| <= i + 1, so only the low bits of the S and M registers are needed.
            s_controls = s_qubits[: i.bit_length()]
            m_controls = m_qubits[: (i + 1).bit_length() + 1]
            angles = _cg_angles(len(s_controls), len(m_controls), max_two_s=i)
            circuit.append(
                CircuitInstruction(UCRYGate(angles), (q, *s_controls, *m_controls))
            )
            # Update S to its outgoing value, which is at most i + 1, adding 1 for a
            # Yamanouchi bit of 0 and subtracting 1 for a Yamanouchi bit of 1
            s_bits = s_qubits[: (i + 1).bit_length()]
            circuit.append(XGate(), (q,))
            for instruction in _controlled_increment(q, s_bits):
                circuit.append(instruction)
            circuit.append(XGate(), (q,))
            for instruction in _controlled_decrement(q, s_bits):
                circuit.append(instruction)

        self.definition = circuit
