import numpy as np
import scipy as sp
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms.tebd import TEBDEngine
from tenpy.linalg.charges import LegPipe
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfFermionSite

from ffsim.linalg import givens_decomposition
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
    r"""The Givens rotation gate.

    The Givens rotation gate as defined in
    `apply_givens_rotation <https://qiskit-community.github.io/ffsim/api/ffsim.html#ffsim.apply_givens_rotation>`__,
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

    # define conjugate phase
    if conj:
        beta = phi + np.pi / 2
        beta = -beta
        phi = beta - np.pi / 2

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
        Ggate_a = (
            np.kron(sp.linalg.expm(1j * phi * Nu), Id)
            @ sp.linalg.expm(
                theta * (np.kron(Cdu @ JW, Cu @ JWu) - np.kron(Cu @ JW, Cdu @ JWu))
            )
            @ np.kron(sp.linalg.expm(-1j * phi * Nu), Id)
        )

    # beta sector / down spins
    if spin in [Spin.BETA, Spin.ALPHA_AND_BETA]:
        Cdd = shfs_nosym.get_op("Cdd").to_ndarray()
        Cd = shfs_nosym.get_op("Cd").to_ndarray()
        JWd = shfs_nosym.get_op("JWd").to_ndarray()
        Nd = shfs_nosym.get_op("Nd").to_ndarray()
        #
        Ggate_b = (
            np.kron(sp.linalg.expm(1j * phi * Nd), Id)
            @ sp.linalg.expm(
                theta * (np.kron(Cdd @ JW, Cd @ JWd) - np.kron(Cd @ JW, Cdd @ JWd))
            )
            @ np.kron(sp.linalg.expm(-1j * phi * Nd), Id)
        )

    # define total gate
    if spin is Spin.ALPHA:
        Ggate = Ggate_a
    elif spin is Spin.BETA:
        Ggate = Ggate_b
    elif spin is Spin.ALPHA_AND_BETA:
        Ggate = Ggate_a @ Ggate_b
    else:
        raise ValueError("undefined spin")

    # convert to (N, Sz)-symmetry-conserved basis
    Ggate_sym = sym_cons_basis(Ggate)

    return Ggate_sym


def num_interaction(theta: float, spin: Spin) -> np.ndarray:
    r"""The number interaction gate.

    The number interaction gate as defined in
    `apply_num_interaction <https://qiskit-community.github.io/ffsim/api/ffsim.html#ffsim.apply_num_interaction>`__,
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

    # alpha sector / up spins
    if spin in [Spin.ALPHA, Spin.ALPHA_AND_BETA]:
        Nu = shfs_nosym.get_op("Nu").to_ndarray()
        Ngate_a = sp.linalg.expm(1j * theta * Nu)

    # beta sector / down spins
    if spin in [Spin.BETA, Spin.ALPHA_AND_BETA]:
        Nd = shfs_nosym.get_op("Nd").to_ndarray()
        Ngate_b = sp.linalg.expm(1j * theta * Nd)

    # define total gate
    if spin is Spin.ALPHA:
        Ngate = Ngate_a
    elif spin is Spin.BETA:
        Ngate = Ngate_b
    elif spin is Spin.ALPHA_AND_BETA:
        Ngate = Ngate_a @ Ngate_b
    else:
        raise ValueError("undefined spin")

    # convert to (N, Sz)-symmetry-conserved basis
    Ngate_sym = sym_cons_basis(Ngate)

    return Ngate_sym


def on_site_interaction(theta: float) -> np.ndarray:
    r"""The on-site interaction gate.

    The on-site interaction gate as defined in
    `apply_on_site_interaction <https://qiskit-community.github.io/ffsim/api/ffsim.html#ffsim.apply_on_site_interaction>`__,
    returned in the TeNPy (N, Sz)-symmetry-conserved basis.

    Args:
        theta: The rotation angle.

    Returns:
        The on-site interaction gate in the TeNPy (N, Sz)-symmetry-conserved basis.
    """

    # define operators
    Nu = shfs_nosym.get_op("Nu").to_ndarray()
    Nd = shfs_nosym.get_op("Nd").to_ndarray()

    # define total gate
    OSgate = sp.linalg.expm(1j * theta * Nu @ Nd)

    # convert to (N, Sz)-symmetry-conserved basis
    OSgate_sym = sym_cons_basis(OSgate)

    return OSgate_sym


def num_num_interaction(theta: float, spin: Spin) -> np.ndarray:
    r"""The number-number interaction gate.

    The number-number interaction gate as defined in
    `apply_num_num_interaction <https://qiskit-community.github.io/ffsim/api/ffsim.html#ffsim.apply_num_num_interaction>`__,
    returned in the TeNPy (N, Sz)-symmetry-conserved basis.

    Args:
        theta: The rotation angle.
        spin: Choice of spin sector(s) to act on.

            - To act on only spin alpha, pass :const:`ffsim.Spin.ALPHA`.
            - To act on only spin beta, pass :const:`ffsim.Spin.BETA`.
            - To act on both spin alpha and spin beta, pass
              :const:`ffsim.Spin.ALPHA_AND_BETA` (this is the default value).

    Returns:
        The number-number interaction gate in the TeNPy (N, Sz)-symmetry-conserved
        basis.
    """

    # alpha sector / up spins
    if spin in [Spin.ALPHA, Spin.ALPHA_AND_BETA]:
        Nu = shfs_nosym.get_op("Nu").to_ndarray()
        NNgate_a = sp.linalg.expm(1j * theta * np.kron(Nu, Nu))

    # beta sector / down spins
    if spin in [Spin.BETA, Spin.ALPHA_AND_BETA]:
        Nd = shfs_nosym.get_op("Nd").to_ndarray()
        NNgate_b = sp.linalg.expm(1j * theta * np.kron(Nd, Nd))

    # define total gate
    if spin is Spin.ALPHA:
        NNgate = NNgate_a
    elif spin is Spin.BETA:
        NNgate = NNgate_b
    elif spin is Spin.ALPHA_AND_BETA:
        NNgate = NNgate_a @ NNgate_b
    else:
        raise ValueError("undefined spin")

    # convert to (N, Sz)-symmetry-conserved basis
    NNgate_sym = sym_cons_basis(NNgate)

    return NNgate_sym


def apply_gate1(U1: np.ndarray, site: int, psi: MPS) -> None:
    r"""Apply a single-site gate to a
    `TeNPy MPS <https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.mps.MPS.html#tenpy.networks.mps.MPS>`__
    wavefunction.

    Args:
        U1: The single-site quantum gate.
        site: The gate will be applied to `site` on the
            `TeNPy MPS <https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.mps.MPS.html#tenpy.networks.mps.MPS>`__
            wavefunction.
        psi: The `TeNPy MPS <https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.mps.MPS.html#tenpy.networks.mps.MPS>`__
            wavefunction.

    Returns:
        None
    """

    # on-site
    U1_npc = npc.Array.from_ndarray(U1, [shfs.leg, shfs.leg.conj()], labels=["p", "p*"])
    psi.apply_local_op(site, U1_npc)


def apply_gate2(
    U2: np.ndarray,
    site: int,
    psi: MPS,
    eng: TEBDEngine,
    chi_list: list,
    norm_tol: float = 1e-5,
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


def apply_orbital_rotation(
    mat: np.ndarray,
    psi: MPS,
    eng: TEBDEngine,
    chi_list: list,
    norm_tol: float = 1e-5,
) -> None:
    r"""Apply an orbital rotation gate to a
    `TeNPy MPS <https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.mps.MPS.html#tenpy.networks.mps.MPS>`__
    wavefunction.

    The orbital rotation gate is defined in
    `apply_orbital_rotation <https://qiskit-community.github.io/ffsim/api/ffsim.html#ffsim.apply_orbital_rotation>`__.

    Args:
        mat: The orbital rotation matrix of dimension `(norb, norb)`.
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

    # Givens decomposition
    givens_list, diag_mat = givens_decomposition(mat)

    # apply the Givens rotation gates
    for gate in givens_list:
        theta = np.arccos(gate.c)
        s = np.conj(gate.s)
        phi = np.real(1j * np.log(-s / np.sin(theta))) if theta else 0
        conj = True if gate.j < gate.i else False
        apply_gate2(
            givens_rotation(theta, Spin.ALPHA_AND_BETA, conj, phi=phi),
            max(gate.i, gate.j),
            psi,
            eng,
            chi_list,
            norm_tol,
        )

    # apply the number interaction gates
    for i, z in enumerate(diag_mat):
        theta = float(np.angle(z))
        apply_gate1(
            np.exp(1j * theta) * num_interaction(-theta, Spin.ALPHA_AND_BETA), i, psi
        )


def apply_diag_coulomb_evolution(
    mat: np.ndarray, psi: MPS, eng: TEBDEngine, chi_list: list, norm_tol: float = 1e-5
) -> None:
    r"""Apply a diagonal Coulomb evolution gate to a
    `TeNPy MPS <https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.mps.MPS.html#tenpy.networks.mps.MPS>`__
    wavefunction.

    The diagonal Coulomb evolution gate is defined in
    `apply_diag_coulomb_evolution <https://qiskit-community.github.io/ffsim/api/ffsim.html#ffsim.apply_diag_coulomb_evolution>`__.

    Args:
        mat: The diagonal Coulomb matrices of dimension `(2, norb, norb)`.
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

    # extract norb
    assert np.shape(mat)[1] == np.shape(mat)[2]
    norb = np.shape(mat)[1]

    # unpack alpha-alpha and alpha-beta matrices
    mat_aa, mat_ab = mat

    # apply alpha-alpha gates
    for i in range(norb):
        for j in range(norb):
            if j > i and mat_aa[i, j]:
                apply_gate2(
                    num_num_interaction(-mat_aa[i, j], Spin.ALPHA_AND_BETA),
                    j,
                    psi,
                    eng,
                    chi_list,
                    norm_tol,
                )

    # apply alpha-beta gates
    for i in range(norb):
        apply_gate1(on_site_interaction(-mat_ab[i, i]), i, psi)
