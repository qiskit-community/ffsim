# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Fermi-Hubbard operator."""

from __future__ import annotations

from math import comb

import numpy as np
import scipy

import ffsim
from ffsim.operators.fermi_hubbard import fermi_hubbard


def test_non_interacting_FH_operator():
    """Test non-interacting Fermi-Hubbard operator."""

    # open boundary conditions
    op = fermi_hubbard(norb=4, t_hop=1)
    ham = ffsim.linear_operator(op, norb=4, nelec=(2, 2))
    eigs, _ = scipy.sparse.linalg.eigsh(ham, which="SA", k=1)
    E = -4.472135955000
    assert np.isclose(eigs[0], E)

    # periodic boundary conditions
    op_PBC = fermi_hubbard(norb=4, t_hop=1, PBC=True)
    ham_PBC = ffsim.linear_operator(op_PBC, norb=4, nelec=(2, 2))
    eigs_PBC, _ = scipy.sparse.linalg.eigsh(ham_PBC, which="SA", k=1)
    E_PBC = -4.000000000000
    assert np.isclose(eigs_PBC[0], E_PBC)


def test_FH_operator_with_U_int():
    """Test Fermi-Hubbard operator with U_int."""

    # open boundary conditions
    op = fermi_hubbard(norb=4, t_hop=1, U_int=2)
    ham = ffsim.linear_operator(op, norb=4, nelec=(2, 2))
    eigs, _ = scipy.sparse.linalg.eigsh(ham, which="SA", k=1)
    E = -2.875942809005
    assert np.isclose(eigs[0], E)

    # periodic boundary conditions
    op_PBC = fermi_hubbard(norb=4, t_hop=1, U_int=2, PBC=True)
    ham_PBC = ffsim.linear_operator(op_PBC, norb=4, nelec=(2, 2))
    eigs_PBC, _ = scipy.sparse.linalg.eigsh(ham_PBC, which="SA", k=1)
    E_PBC = -2.828427124746
    assert np.isclose(eigs_PBC[0], E_PBC)


def test_FH_operator_with_U_int_mu_pot():
    """Test Fermi-Hubbard operator with U_int and mu_pot."""

    # open boundary conditions
    op = fermi_hubbard(norb=4, t_hop=1, U_int=2, mu_pot=3)
    ham = ffsim.linear_operator(op, norb=4, nelec=(2, 2))
    eigs, _ = scipy.sparse.linalg.eigsh(ham, which="SA", k=1)
    E = -14.875942809005
    assert np.isclose(eigs[0], E)

    # periodic boundary conditions
    op_PBC = fermi_hubbard(norb=4, t_hop=1, U_int=2, mu_pot=3, PBC=True)
    ham_PBC = ffsim.linear_operator(op_PBC, norb=4, nelec=(2, 2))
    eigs_PBC, _ = scipy.sparse.linalg.eigsh(ham_PBC, which="SA", k=1)
    E_PBC = -14.828427124746
    assert np.isclose(eigs_PBC[0], E_PBC)


def test_FH_operator_with_U_int_mu_pot_V_int():
    """Test Fermi-Hubbard operator with U_int, mu_pot, and V_int."""

    # open boundary conditions
    op = fermi_hubbard(norb=4, t_hop=1, U_int=2, mu_pot=3, V_int=4)
    ham = ffsim.linear_operator(op, norb=4, nelec=(2, 2))
    eigs, _ = scipy.sparse.linalg.eigsh(ham, which="SA", k=1)
    E = -9.961978205599
    assert np.isclose(eigs[0], E)

    # periodic boundary conditions
    op_PBC = fermi_hubbard(norb=4, t_hop=1, U_int=2, mu_pot=3, V_int=4, PBC=True)
    ham_PBC = ffsim.linear_operator(op_PBC, norb=4, nelec=(2, 2))
    eigs_PBC, _ = scipy.sparse.linalg.eigsh(ham_PBC, which="SA", k=1)
    E_PBC = -8.781962448006
    assert np.isclose(eigs_PBC[0], E_PBC)


def test_FH_operator_with_unequal_filling():
    """Test Fermi-Hubbard operator with unequal filling."""

    # open boundary conditions
    op = fermi_hubbard(norb=4, t_hop=1, U_int=2, mu_pot=3, V_int=4)
    ham = ffsim.linear_operator(op, norb=4, nelec=(1, 3))
    eigs, _ = scipy.sparse.linalg.eigsh(ham, which="SA", k=1)
    E = -6.615276287167
    assert np.isclose(eigs[0], E)

    # periodic boundary conditions
    op_PBC = fermi_hubbard(norb=4, t_hop=1, U_int=2, mu_pot=3, V_int=4, PBC=True)
    ham_PBC = ffsim.linear_operator(op_PBC, norb=4, nelec=(1, 3))
    eigs_PBC, _ = scipy.sparse.linalg.eigsh(ham_PBC, which="SA", k=1)
    E_PBC = -0.828427124746
    assert np.isclose(eigs_PBC[0], E_PBC)


def test_FH_operator_hermiticity():
    """Test Fermi-Hubbard operator hermiticity."""

    N_orb, N_alpha, N_beta = 4, 3, 1
    dim = comb(N_orb, N_alpha) * comb(N_orb, N_beta)

    # open boundary conditions
    op = fermi_hubbard(norb=N_orb, t_hop=0.1, U_int=0.2, mu_pot=0.3, V_int=0.4)
    ham = ffsim.linear_operator(op, norb=N_orb, nelec=(N_alpha, N_beta))
    assert np.allclose(ham.dot(np.eye(dim)), ham.H.dot(np.eye(dim)))

    # periodic boundary conditions
    op_PBC = fermi_hubbard(
        norb=N_orb, t_hop=0.1, U_int=0.2, mu_pot=0.3, V_int=0.4, PBC=True
    )
    ham_PBC = ffsim.linear_operator(op_PBC, norb=N_orb, nelec=(N_alpha, N_beta))
    assert np.allclose(ham_PBC.dot(np.eye(dim)), ham_PBC.H.dot(np.eye(dim)))
