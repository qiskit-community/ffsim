import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from tenpy.algorithms.tebd import TEBDEngine

import ffsim
from ffsim.tenpy.circuits.gates import (
    CPhase,
    CPhase_onsite,
    Phase,
    XXPlusYY,
    gate1,
    gate2,
)
from ffsim.tenpy.util import product_state_as_mps

# from tenpy.models.hubbard import FermiHubbardChain


def lucj_circuit_as_mps(norb, nelec, lucj_operator, options, norm_tol=1e-5):
    r"""Return the LUCJ circuit as an MPS.

    Args:
        norb: The number of spatial orbitals.
        nelec: Either a single integer representing the number of fermions for a
            spinless system, or a pair of integers storing the numbers of spin alpha
            and spin beta fermions.
        lucj_operator: The LUCJ operator.
        options: The parameters passed to the TEBDEngine, as defined in the
            TeNPy documentation.
        norm_tol: The norm error above which we recanonicalize the wavefunction, as
            defined in the TeNPy documentation.

    Return type:
        `TeNPy MPS <https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.mps.MPS.html#tenpy.networks.mps.MPS>`__

    Returns:
        The LUCJ circuit as an MPS.
    """

    # initialize chi_list
    chi_list = []

    # prepare initial Hartree-Fock state
    psi = product_state_as_mps(norb, nelec, 0)

    # construct the qiskit circuit
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(lucj_operator), qubits)

    # define the TEBD engine
    eng = TEBDEngine(psi, None, options)

    # execute the tenpy circuit
    for ins in circuit.decompose(reps=2):
        if ins.operation.name == "p":
            qubit = ins.qubits[0]
            idx = qubit._index
            spin_flag = "up" if idx < norb else "down"
            lmbda = ins.operation.params[0]
            gate1(Phase(spin_flag, lmbda), idx % norb, psi)
        elif ins.operation.name == "xx_plus_yy":
            qubit0 = ins.qubits[0]
            qubit1 = ins.qubits[1]
            idx0, idx1 = qubit0._index, qubit1._index
            if idx0 < norb and idx1 < norb:
                spin_flag = "up"
            elif idx0 >= norb and idx1 >= norb:
                spin_flag = "down"
            else:
                raise ValueError("XXPlusYY gate not allowed across spin sectors")
            theta_val = ins.operation.params[0]
            beta_val = ins.operation.params[1]
            # directionality important when beta!=0
            conj_flag = True if idx0 > idx1 else False
            gate2(
                XXPlusYY(spin_flag, theta_val, beta_val, conj_flag),
                max(idx0 % norb, idx1 % norb),
                psi,
                eng,
                chi_list,
                norm_tol,
            )
        elif ins.operation.name == "cp":
            qubit0 = ins.qubits[0]
            qubit1 = ins.qubits[1]
            idx0, idx1 = qubit0._index, qubit1._index
            lmbda = ins.operation.params[0]
            # onsite (different spins)
            if np.abs(idx0 - idx1) == norb:
                gate1(CPhase_onsite(lmbda), min(idx0, idx1), psi)
            # NN (up spins)
            elif np.abs(idx0 - idx1) == 1 and idx0 < norb and idx1 < norb:
                gate2(
                    CPhase("up", lmbda), max(idx0, idx1), psi, eng, chi_list, norm_tol
                )
            # NN (down spins)
            elif np.abs(idx0 - idx1) == 1 and idx0 >= norb and idx1 >= norb:
                gate2(
                    CPhase("down", lmbda),
                    max(idx0 % norb, idx1 % norb),
                    psi,
                    eng,
                    chi_list,
                    norm_tol,
                )
            else:
                raise ValueError(
                    "CPhase only implemented onsite (different spins) "
                    "and NN (same spins)"
                )
        else:
            raise ValueError(f"gate {ins.operation.name} not implemented.")

    return psi, chi_list
