import numpy as np
import scipy as sp
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms.tebd import TEBDEngine
from tenpy.linalg.charges import LegPipe
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfFermionSite

from ffsim.spin import Spin

# ignore lowercase argument and variable checks to maintain TeNPy naming conventions
# ruff: noqa: N803, N806

# define sites
shfs_nosym = SpinHalfFermionSite(cons_N=None, cons_Sz=None)
shfs = SpinHalfFermionSite(cons_N="N", cons_Sz="Sz")
shfsc = LegPipe([shfs.leg, shfs.leg])


def sym_cons_basis(gate: np.ndarray) -> np.ndarray:
    r"""Convert a gate to the TeNPy (N, Sz)-symmetry-conserved basis.

    Args:
        gate: The quantum gate.

    Returns:
        The quantum gate in the TeNPy (N, Sz)-symmetry-conserved basis.
    """

    # convert to (N, Sz)-symmetry-conserved basis
    if np.shape(gate) == (4, 4):  # 1-site gate
        swap_list = [1, 3, 0, 2]
    elif np.shape(gate) == (16, 16):  # 2-site gate
        swap_list = [5, 11, 2, 7, 12, 15, 9, 14, 1, 6, 0, 3, 8, 13, 4, 10]
    else:
        raise ValueError(
            "only 1-site and 2-site gates implemented for symmetry basis conversion"
        )

    P = np.zeros(np.shape(gate))
    for i, s in enumerate(swap_list):
        P[i, s] = 1

    gate_sym = P.T @ gate @ P

    return gate_sym


def givens_rotation(
    theta: float, spin: Spin, conj: bool = False, *, phi: float = 0.0
) -> np.ndarray:
    r"""The Givens gate.

    The Givens rotation gate as defined in the
    `ffsim documentation <https://qiskit-community.github.io/ffsim/api/ffsim.html#ffsim.apply_givens_rotation>`__,
    returned in the TeNPy (N, Sz)-symmetry-conserved basis.

    Args:
        theta: The rotation angle.
        spin: Choice of spin sector(s) to act on.

            - To act on only spin alpha, pass :const:`ffsim.Spin.ALPHA`.
            - To act on only spin beta, pass :const:`ffsim.Spin.BETA`.
            - To act on both spin alpha and spin beta, pass
              :const:`ffsim.Spin.ALPHA_AND_BETA`.
        conj: The direction of the gate. By default, we use the little endian
            convention, as in Qiskit.
        phi: The phase angle.

    Returns:
        The Givens rotation gate in the TeNPy (N, Sz)-symmetry-conserved basis.
    """

    # translate angle parameters
    theta = 2 * theta
    phi = phi + np.pi / 2

    # define conjugate operator
    if conj:
        phi = -phi

    # define operators
    Id = shfs_nosym.get_op("Id").to_ndarray()
    JW = shfs_nosym.get_op("JW").to_ndarray()

    # alpha sector / up spins
    if spin in [Spin.ALPHA, Spin.ALPHA_AND_BETA]:
        Cdu = shfs_nosym.get_op("Cdu").to_ndarray()
        Cu = shfs_nosym.get_op("Cu").to_ndarray()
        JWu = shfs_nosym.get_op("JWu").to_ndarray()
        Nu = shfs_nosym.get_op("Nu").to_ndarray()
        #
        Xu1 = (Cdu + Cu) @ JW
        Xu2 = (Cdu + Cu) @ JWu
        Yu1 = -1j * (Cdu - Cu) @ JW
        Yu2 = -1j * (Cdu - Cu) @ JWu
        Zu = 2 * Nu - Id
        RZu0 = np.kron(sp.linalg.expm(-1j * (phi / 2) * Zu), Id)
        #
        XYgate_a = (
            np.conj(RZu0)
            @ sp.linalg.expm(
                -1j * (theta / 4) * (np.kron(Xu1, Xu2) + np.kron(Yu1, Yu2))
            )
            @ RZu0
        )

    # beta sector / down spins
    if spin in [Spin.BETA, Spin.ALPHA_AND_BETA]:
        Cdd = shfs_nosym.get_op("Cdd").to_ndarray()
        Cd = shfs_nosym.get_op("Cd").to_ndarray()
        JWd = shfs_nosym.get_op("JWd").to_ndarray()
        Nd = shfs_nosym.get_op("Nd").to_ndarray()
        #
        Xd1 = (Cdd + Cd) @ JW
        Xd2 = (Cdd + Cd) @ JWd
        Yd1 = -1j * (Cdd - Cd) @ JW
        Yd2 = -1j * (Cdd - Cd) @ JWd
        Zd = 2 * Nd - Id
        RZd0 = np.kron(sp.linalg.expm(-1j * (phi / 2) * Zd), Id)
        #
        XYgate_b = (
            np.conj(RZd0)
            @ sp.linalg.expm(
                -1j * (theta / 4) * (np.kron(Xd1, Xd2) + np.kron(Yd1, Yd2))
            )
            @ RZd0
        )

    # define total gate
    if spin is Spin.ALPHA:
        XYgate = XYgate_a
    elif spin is Spin.BETA:
        XYgate = XYgate_b
    elif spin is Spin.ALPHA_AND_BETA:
        XYgate = XYgate_a @ XYgate_b
    else:
        raise ValueError("undefined spin")

    # convert to (N, Sz)-symmetry-conserved basis
    XYgate_sym = sym_cons_basis(XYgate)

    return XYgate_sym


def xy(spin: str, theta: float, beta: float, conj: bool = False) -> np.ndarray:
    r"""The XXPlusYY gate.

    The XXPlusYY gate as defined in the
    `Qiskit documentation <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.XXPlusYYGate>`__,
    returned in the TeNPy (N, Sz)-symmetry-conserved basis.

    Args:
        spin: The spin sector ("up" or "down").
        theta: The rotation angle.
        beta: The phase angle.
        conj: The direction of the gate. By default, we use the little endian
            convention, as in Qiskit.

    Returns:
        The XXPlusYY gate in the TeNPy (N, Sz)-symmetry-conserved basis.
    """

    # define conjugate operator
    if conj:
        beta = -beta

    # define operators
    Id = shfs_nosym.get_op("Id").to_ndarray()
    JW = shfs_nosym.get_op("JW").to_ndarray()

    if spin == "up":
        # alpha sector / up spins
        Cdu = shfs_nosym.get_op("Cdu").to_ndarray()
        Cu = shfs_nosym.get_op("Cu").to_ndarray()
        JWu = shfs_nosym.get_op("JWu").to_ndarray()
        Nu = shfs_nosym.get_op("Nu").to_ndarray()
        #
        Xu1 = (Cdu + Cu) @ JW
        Xu2 = (Cdu + Cu) @ JWu
        Yu1 = -1j * (Cdu - Cu) @ JW
        Yu2 = -1j * (Cdu - Cu) @ JWu
        Zu = 2 * Nu - Id
        RZu0 = np.kron(sp.linalg.expm(-1j * (beta / 2) * Zu), Id)
        #
        XYgate = (
            np.conj(RZu0)
            @ sp.linalg.expm(
                -1j * (theta / 4) * (np.kron(Xu1, Xu2) + np.kron(Yu1, Yu2))
            )
            @ RZu0
        )
    elif spin == "down":
        # beta sector / down spins
        Cdd = shfs_nosym.get_op("Cdd").to_ndarray()
        Cd = shfs_nosym.get_op("Cd").to_ndarray()
        JWd = shfs_nosym.get_op("JWd").to_ndarray()
        Nd = shfs_nosym.get_op("Nd").to_ndarray()
        #
        Xd1 = (Cdd + Cd) @ JW
        Xd2 = (Cdd + Cd) @ JWd
        Yd1 = -1j * (Cdd - Cd) @ JW
        Yd2 = -1j * (Cdd - Cd) @ JWd
        Zd = 2 * Nd - Id
        RZd0 = np.kron(sp.linalg.expm(-1j * (beta / 2) * Zd), Id)
        #
        XYgate = (
            np.conj(RZd0)
            @ sp.linalg.expm(
                -1j * (theta / 4) * (np.kron(Xd1, Xd2) + np.kron(Yd1, Yd2))
            )
            @ RZd0
        )
    else:
        raise ValueError("undefined spin")

    # convert to (N, Sz)-symmetry-conserved basis
    XYgate_sym = sym_cons_basis(XYgate)

    return XYgate_sym


def num_interaction(theta: float, spin: Spin) -> np.ndarray:
    r"""The number interaction gate.

    The number interaction gate as defined in the
    `ffsim documentation <https://qiskit-community.github.io/ffsim/api/ffsim.html#ffsim.apply_num_interaction>`__,
    returned in the TeNPy (N, Sz)-symmetry-conserved basis.

    Args:
        theta: The rotation angle.
        spin: Choice of spin sector(s) to act on.

            - To act on only spin alpha, pass :const:`ffsim.Spin.ALPHA`.
            - To act on only spin beta, pass :const:`ffsim.Spin.BETA`.
            - To act on both spin alpha and spin beta, pass
              :const:`ffsim.Spin.ALPHA_AND_BETA` (this is the default value).

    Returns:
        The number interaction gate in the TeNPy (N, Sz)-symmetry-conserved basis.
    """

    # define operators
    Id = shfs_nosym.get_op("Id").to_ndarray()

    # alpha sector / up spins
    if spin in [Spin.ALPHA, Spin.ALPHA_AND_BETA]:
        Nu = shfs_nosym.get_op("Nu").to_ndarray()
        Zu = 2 * Nu - Id
        RZu = sp.linalg.expm(-1j * (theta / 2) * Zu)
        Pgate_a = np.exp(1j * (theta / 2)) * RZu

    # beta sector / down spins
    if spin in [Spin.BETA, Spin.ALPHA_AND_BETA]:
        Nd = shfs_nosym.get_op("Nd").to_ndarray()
        Zd = 2 * Nd - Id
        RZd = sp.linalg.expm(-1j * (theta / 2) * Zd)
        Pgate_b = np.exp(1j * (theta / 2)) * RZd

    # define total gate
    if spin is Spin.ALPHA:
        Pgate = Pgate_a
    elif spin is Spin.BETA:
        Pgate = Pgate_b
    elif spin is Spin.ALPHA_AND_BETA:
        Pgate = Pgate_a @ Pgate_b
    else:
        raise ValueError("undefined spin")

    # convert to (N, Sz)-symmetry-conserved basis
    Pgate_sym = sym_cons_basis(Pgate)

    return Pgate_sym


def phase(spin: str, theta: float) -> np.ndarray:
    r"""The Phase gate.

    The Phase gate as defined in the
    `Qiskit documentation <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.PhaseGate>`__,
    returned in the TeNPy (N, Sz)-symmetry-conserved basis.

    Args:
        spin: The spin sector ("up" or "down").
        theta: The rotation angle.

    Returns:
        The Phase gate in the TeNPy (N, Sz)-symmetry-conserved basis.
    """

    # define operators
    Id = shfs_nosym.get_op("Id").to_ndarray()

    if spin == "up":
        # alpha sector / up spins
        Nu = shfs_nosym.get_op("Nu").to_ndarray()
        Zu = 2 * Nu - Id
        RZu = sp.linalg.expm(-1j * (theta / 2) * Zu)
        Pgate = np.exp(1j * (theta / 2)) * RZu
    elif spin == "down":
        # beta sector / down spins
        Nd = shfs_nosym.get_op("Nd").to_ndarray()
        Zd = 2 * Nd - Id
        RZd = sp.linalg.expm(-1j * (theta / 2) * Zd)
        Pgate = np.exp(1j * (theta / 2)) * RZd
    else:
        raise ValueError("undefined spin")

    # convert to (N, Sz)-symmetry-conserved basis
    Pgate_sym = sym_cons_basis(Pgate)

    return Pgate_sym


def on_site_interaction(theta: float) -> np.ndarray:
    r"""The on-site interaction gate.

    The on-site interaction gate as defined in the
    `ffsim documentation <https://qiskit-community.github.io/ffsim/api/ffsim.html#ffsim.apply_on_site_interaction>`__,
    returned in the TeNPy (N, Sz)-symmetry-conserved basis.

    Args:
        theta: The rotation angle.

    Returns:
        The on-site interaction gate in the TeNPy (N, Sz)-symmetry-conserved basis.
    """

    CPgate = np.eye(4, dtype=complex)
    CPgate[3, 3] = np.exp(-1j * theta)  # minus sign

    # convert to (N, Sz)-symmetry-conserved basis
    CPgate_sym = sym_cons_basis(CPgate)

    return CPgate_sym


def cphase1(theta: float) -> np.ndarray:
    r"""The single-site CPhase gate.

    The single-site CPhase gate as defined in the
    `Qiskit documentation <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.CPhaseGate>`__,
    returned in the TeNPy (N, Sz)-symmetry-conserved basis.

    .. note::
        A two-site CPhase gate in the qubit basis may translate to a single-site CPhase
        gate in the fermion basis.

    Args:
        theta: The rotation angle.

    Returns:
        The single-site CPhase gate in the TeNPy (N, Sz)-symmetry-conserved basis.
    """

    CPgate = np.eye(4, dtype=complex)
    CPgate[3, 3] = np.exp(-1j * theta)  # minus sign

    # convert to (N, Sz)-symmetry-conserved basis
    CPgate_sym = sym_cons_basis(CPgate)

    return CPgate_sym


def cphase2(spin: str, theta: float) -> np.ndarray:
    r"""The two-site CPhase gate.

    The two-site CPhase gate as defined in the
    `Qiskit documentation <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.XXPlusYYGate>`__,
    returned in the TeNPy (N, Sz)-symmetry-conserved basis.

    .. note::
        A two-site CPhase gate in the qubit basis may translate to a single-site CPhase
        gate in the fermion basis.

    Args:
        spin: The spin sector ("up" or "down").
        theta: The rotation angle.

    Returns:
        The two-site CPhase gate in the TeNPy (N, Sz)-symmetry-conserved basis.
    """

    # define operators
    Id = shfs_nosym.get_op("Id").to_ndarray()

    state_0 = np.array([1, 0, 0, 0])
    state_1 = np.array([0, 1, 0, 0])
    state_2 = np.array([0, 0, 1, 0])
    state_3 = np.array([0, 0, 0, 1])

    outer_0 = np.outer(state_0, state_0)
    outer_1 = np.outer(state_1, state_1)
    outer_2 = np.outer(state_2, state_2)
    outer_3 = np.outer(state_3, state_3)

    if spin == "up":
        # alpha sector / up spins
        Nu = shfs_nosym.get_op("Nu").to_ndarray()
        Zu = 2 * Nu - Id
        RZu = sp.linalg.expm(-1j * (theta / 2) * Zu)
        Pup = np.exp(-1j * (theta / 2)) * RZu  # minus sign
        CPgate = (
            np.kron(outer_0, Id)
            + np.kron(outer_1, Pup)
            + np.kron(outer_2, Id)
            + np.kron(outer_3, Pup)
        )
    elif spin == "down":
        # beta sector / down spins
        Nd = shfs_nosym.get_op("Nd").to_ndarray()
        Zd = 2 * Nd - Id
        RZd = sp.linalg.expm(-1j * (theta / 2) * Zd)
        Pdw = np.exp(-1j * (theta / 2)) * RZd  # minus sign
        CPgate = (
            np.kron(outer_0, Id)
            + np.kron(outer_1, Id)
            + np.kron(outer_2, Pdw)
            + np.kron(outer_3, Pdw)
        )
    else:
        raise ValueError("undefined spin")

    # convert to (N, Sz)-symmetry-conserved basis
    CPgate_sym = sym_cons_basis(CPgate)

    return CPgate_sym


def gate1(U1: np.ndarray, site: int, psi: MPS) -> None:
    r"""Apply a single-site gate to a
    `TeNPy MPS <https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.mps.MPS.html#tenpy.networks.mps.MPS>`__
    wavefunction.

    Args:
        U1: The single-site quantum gate.
        site: The gate will be applied to `site` on the
            `TeNPy MPS <https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.mps.MPS.html#tenpy.networks.mps.MPS>`__
            wavefunction.
        psi: The wavefunction MPS.

    Returns:
        None
    """

    # on-site
    U1_npc = npc.Array.from_ndarray(U1, [shfs.leg, shfs.leg.conj()], labels=["p", "p*"])
    psi.apply_local_op(site, U1_npc)


def gate2(
    U2: np.ndarray,
    site: int,
    psi: MPS,
    eng: TEBDEngine,
    chi_list: list,
    norm_tol: float,
) -> None:
    r"""Apply a two-site gate to a `TeNPy MPS <https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.mps.MPS.html#tenpy.networks.mps.MPS>`__
    wavefunction.

    Args:
        U2: The two-site quantum gate.
        site: The gate will be applied to `(site-1, site)` on the `TeNPy MPS <https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.mps.MPS.html#tenpy.networks.mps.MPS>`__
            wavefunction.
        psi: The `TeNPy MPS <https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.mps.MPS.html#tenpy.networks.mps.MPS>`__
            wavefunction.
        eng: The
            `TeNPy TEBDEngine <https://tenpy.readthedocs.io/en/latest/reference/tenpy.algorithms.tebd.TEBDEngine.html#tenpy.algorithms.tebd.TEBDEngine>`__.
        chi_list: The list to which to append the MPS bond dimensions as the circuit is
            evaluated.
        norm_tol: The norm error above which we recanonicalize the wavefunction, as
            defined in the
            `TeNPy documentation <https://tenpy.readthedocs.io/en/latest/reference/tenpy.algorithms.dmrg.DMRGEngine.html#cfg-option-DMRGEngine.norm_tol>`__.

    Returns:
        None
    """

    # bond between (site-1, site)
    U2_npc = npc.Array.from_ndarray(
        U2, [shfsc, shfsc.conj()], labels=["(p0.p1)", "(p0*.p1*)"]
    )
    U2_npc_split = U2_npc.split_legs()
    eng.update_bond(site, U2_npc_split)
    chi_list.append(psi.chi)

    # recanonicalize psi if below error threshold
    if np.linalg.norm(psi.norm_test()) > norm_tol:
        psi.canonical_form_finite()
