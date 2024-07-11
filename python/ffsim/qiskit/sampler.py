# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""ffsim's implementation of the Qiskit Sampler primitive."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.primitives import (
    BaseSamplerV2,
    BitArray,
    DataBin,
    PrimitiveJob,
    PrimitiveResult,
    SamplerPubLike,
    SamplerPubResult,
)
from qiskit.primitives.containers.sampler_pub import SamplerPub

from ffsim import states
from ffsim.qiskit.sim import final_state_vector


class FfsimSampler(BaseSamplerV2):
    """Implementation of the Qiskit Sampler primitive backed by ffsim."""

    def __init__(
        self,
        *,
        default_shots: int = 1024,
        seed: np.random.Generator | int | None = None,
    ):
        """Initialize the ffsim sampler.

        Args:
            default_shots: The default shots to use if not specified during run.
            seed: A seed to initialize the pseudorandom number generator.
                Should be a valid input to ``np.random.default_rng``.
        """
        self._default_shots = default_shots
        self._rng = np.random.default_rng(seed)

    def run(
        self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None
    ) -> PrimitiveJob[PrimitiveResult[SamplerPubResult]]:
        if shots is None:
            shots = self._default_shots
        coerced_pubs = [SamplerPub.coerce(pub, shots) for pub in pubs]
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _run(self, pubs: Iterable[SamplerPub]) -> PrimitiveResult[SamplerPubResult]:
        results = [self._run_pub(pub) for pub in pubs]
        return PrimitiveResult(results)

    def _run_pub(self, pub: SamplerPub) -> SamplerPubResult:
        circuit, qargs, meas_info = _preprocess_circuit(pub.circuit)
        bound_circuits = pub.parameter_values.bind_all(circuit)
        arrays = {
            item.creg_name: np.zeros(
                bound_circuits.shape + (pub.shots, item.num_bytes), dtype=np.uint8
            )
            for item in meas_info
        }
        for index, bound_circuit in np.ndenumerate(bound_circuits):
            final_state = final_state_vector(bound_circuit)
            if qargs:
                norb, nelec = final_state.norb, final_state.nelec
                if isinstance(nelec, int):
                    orbs = qargs
                else:
                    orbs_a = [q for q in qargs if q < norb]
                    orbs_b = [q % norb for q in qargs if q >= norb]
                    orbs = (orbs_a, orbs_b)
                samples = states.sample_state_vector(
                    final_state, orbs=orbs, shots=pub.shots, seed=self._rng
                )
            else:
                samples = [""] * pub.shots
            samples_array = np.array(
                [np.fromiter(sample, dtype=np.uint8) for sample in samples]
            )
            for item in meas_info:
                ary = _samples_to_packed_array(
                    samples_array, item.num_bits, item.qreg_indices
                )
                arrays[item.creg_name][index] = ary

        meas = {
            item.creg_name: BitArray(arrays[item.creg_name], item.num_bits)
            for item in meas_info
        }
        return SamplerPubResult(
            DataBin(**meas, shape=pub.shape), metadata={"shots": pub.shots}
        )


@dataclass
class _MeasureInfo:
    creg_name: str
    num_bits: int
    num_bytes: int
    qreg_indices: list[int]


def _min_num_bytes(num_bits: int) -> int:
    """Return the minimum number of bytes needed to store ``num_bits``."""
    return num_bits // 8 + (num_bits % 8 > 0)


def _preprocess_circuit(circuit: QuantumCircuit):
    num_bits_dict = {creg.name: creg.size for creg in circuit.cregs}
    mapping = _final_measurement_mapping(circuit)
    qargs = sorted(set(mapping.values()))
    qargs_index = {v: k for k, v in enumerate(qargs)}
    circuit = circuit.remove_final_measurements(inplace=False)
    # num_qubits is used as sentinel to fill 0 in _samples_to_packed_array
    sentinel = len(qargs)
    indices = {key: [sentinel] * val for key, val in num_bits_dict.items()}
    for key, qreg in mapping.items():
        creg, ind = key
        indices[creg.name][ind] = qargs_index[qreg]
    meas_info = [
        _MeasureInfo(
            creg_name=name,
            num_bits=num_bits,
            num_bytes=_min_num_bytes(num_bits),
            qreg_indices=indices[name],
        )
        for name, num_bits in num_bits_dict.items()
    ]
    return circuit, qargs, meas_info


def _final_measurement_mapping(
    circuit: QuantumCircuit,
) -> dict[tuple[ClassicalRegister, int], int]:
    """Return the final measurement mapping for the circuit.

    Parameters:
        circuit: Input quantum circuit.

    Returns:
        Mapping of classical bits to qubits for final measurements.
    """
    active_qubits = set(range(circuit.num_qubits))
    active_cbits = set(range(circuit.num_clbits))

    # Find final measurements starting in back
    mapping = {}
    for item in circuit[::-1]:
        if item.operation.name == "measure":
            loc = circuit.find_bit(item.clbits[0])
            cbit = loc.index
            qbit = circuit.find_bit(item.qubits[0]).index
            if cbit in active_cbits and qbit in active_qubits:
                for creg in loc.registers:
                    mapping[creg] = qbit
                active_cbits.remove(cbit)
        elif item.operation.name not in ["barrier", "delay"]:
            for qq in item.qubits:
                _temp_qubit = circuit.find_bit(qq).index
                if _temp_qubit in active_qubits:
                    active_qubits.remove(_temp_qubit)

        if not active_cbits or not active_qubits:
            break

    return mapping


def _samples_to_packed_array(
    samples: NDArray[np.uint8], num_bits: int, indices: list[int]
) -> NDArray[np.uint8]:
    # samples of `Statevector.sample_memory` will be in the order of
    # qubit_last, ..., qubit_1, qubit_0.
    # reverse the sample order into qubit_0, qubit_1, ..., qubit_last and
    # pad 0 in the rightmost to be used for the sentinel introduced by
    # _preprocess_circuit.
    ary = np.pad(samples[:, ::-1], ((0, 0), (0, 1)), constant_values=0)
    # place samples in the order of clbit_last, ..., clbit_1, clbit_0
    ary = ary[:, indices[::-1]]
    # pad 0 in the left to align the number to be mod 8
    # since np.packbits(bitorder='big') pads 0 to the right.
    pad_size = -num_bits % 8
    ary = np.pad(ary, ((0, 0), (pad_size, 0)), constant_values=0)
    # pack bits in big endian order
    ary = np.packbits(ary, axis=-1)
    return ary
