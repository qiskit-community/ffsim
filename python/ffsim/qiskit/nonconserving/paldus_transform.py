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
from qiskit.circuit.library import CXGate, CZGate, UCRYGate

from ffsim.qiskit.nonconserving._util import (
    _cg_angles,
    _controlled_decrement,
    _controlled_increment,
)


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

        for i, (a, b) in enumerate(zip(alpha_qubits, beta_qubits)):
            # Update M to its outgoing value. All bits are updated to keep the
            # two's complement sign extension correct.
            for instruction in _controlled_increment(a, m_qubits):
                circuit.append(instruction)
            for instruction in _controlled_decrement(b, m_qubits):
                circuit.append(instruction)
            # Rotate by Clebsch-Gordan coefficients. Before this step, 2S <= i and
            # |2M| <= i + 1, so only the low bits of the S and M registers are needed.
            for instruction in _cg_rotation(
                a,
                b,
                s_qubits[: i.bit_length()],
                m_qubits[: (i + 1).bit_length() + 1],
                max_two_s=i,
            ):
                circuit.append(instruction)
            # Update S to its outgoing value, which is at most i + 1
            s_bits = s_qubits[: (i + 1).bit_length()]
            for instruction in _controlled_increment(a, s_bits):
                circuit.append(instruction)
            for instruction in _controlled_decrement(b, s_bits):
                circuit.append(instruction)
            # Update N to its outgoing value, which is at most 2 * (i + 1)
            n_bits = n_qubits[: (2 * (i + 1)).bit_length()]
            for instruction in _controlled_increment(a, n_bits):
                circuit.append(instruction)
            for instruction in _controlled_increment(b, n_bits):
                circuit.append(instruction)

        self.definition = circuit


def _cg_rotation(
    alpha: Qubit,
    beta: Qubit,
    s_qubits: Sequence[Qubit],
    m_qubits: Sequence[Qubit],
    max_two_s: int,
) -> Iterator[CircuitInstruction]:
    r"""Givens rotation on an orbital multiplexed on the S and M registers.

    For incoming spin S and outgoing spin projection M, maps

    .. math::
        |01\rangle \mapsto c |01\rangle + s |10\rangle, \quad
        |10\rangle \mapsto -s |01\rangle + c |10\rangle,

    where :math:`c = \sqrt{(S + M + 1/2) / (2S + 1)}` and
    :math:`s = \sqrt{(S - M + 1/2) / (2S + 1)}` are Clebsch-Gordan coefficients.
    Here the bit strings list the alpha qubit first. The registers hold 2S and 2M
    (in two's complement) and may be truncated to their low bits, as long as they
    can represent all 2S up to max_two_s and all 2M with |2M| <= max_two_s + 1.
    """
    # The rotation is conjugated by CX gates that set alpha to 1 for inputs 01 and
    # 10, and to 0 for inputs 00 and 11. Alpha is the least significant control.
    angles = [0.0] * (1 << (1 + len(s_qubits) + len(m_qubits)))
    for index, angle in enumerate(_cg_angles(len(s_qubits), len(m_qubits), max_two_s)):
        angles[1 | (index << 1)] = angle
    yield CircuitInstruction(CXGate(), (beta, alpha))
    yield CircuitInstruction(UCRYGate(angles), (beta, alpha, *s_qubits, *m_qubits))
    yield CircuitInstruction(CXGate(), (beta, alpha))
