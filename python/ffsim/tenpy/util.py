# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""TeNPy utility functions."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

import numpy as np
import tenpy.linalg.np_conserved as npc
from numpy.typing import NDArray
from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
from tenpy.networks.site import FermionSite, SpinHalfFermionSite

import ffsim


def mps_to_statevector(mps: MPS) -> NDArray[np.complex128]:
    r"""Return the MPS as a state vector.

    Args:
        mps: The MPS.

    Returns:
        The state vector.
    """

    # generate the ffsim-ordered list of product states
    norb = mps.L
    n_alpha = round(np.sum(mps.expectation_value("Nu")))
    n_beta = round(np.sum(mps.expectation_value("Nd")))
    product_states = _generate_product_states(norb, (n_alpha, n_beta))

    # initialize the TeNPy ExactDiag class instance
    charge_sector = mps.get_total_charge(True)
    exact_diag = ExactDiag(FermiHubbardChain({"L": norb}), charge_sector=charge_sector)

    # determine the mapping from TeNPy basis to ffsim basis
    basis_ordering_ffsim, swap_factors_ffsim = _map_tenpy_to_ffsim_basis(
        product_states, exact_diag
    )

    # convert TeNPy MPS to ffsim statevector
    statevector = cast(NDArray[np.complex128], exact_diag.mps_to_full(mps).to_ndarray())
    statevector = np.multiply(swap_factors_ffsim, statevector[basis_ordering_ffsim])

    return statevector


def statevector_to_mps(
    statevector: NDArray[np.complex128],
    norb: int,
    nelec: tuple[int, int],
) -> MPS:
    r"""Return the state vector as an MPS.

    Args:
        statevector: The state vector.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.

    Returns:
        The MPS.
    """

    # check if state vector is basis state
    basis_state = np.count_nonzero(statevector) == 1

    # generate the ffsim-ordered list of product states
    if basis_state:
        idx = int(np.flatnonzero(statevector)[0])
        string = ffsim.addresses_to_strings(
            [idx],
            norb,
            nelec,
            concatenate=False,
            bitstring_type=ffsim.BitstringType.INT,
        )
        bitstring = (string[0][0], string[1][0])
        product_states = [_bitstring_to_product_state(bitstring, norb)]
    else:
        product_states = _generate_product_states(norb, nelec)

    # construct the reference product state MPS
    shfs = SpinHalfFermionSite(cons_N="N", cons_Sz="Sz")
    mps_reference = MPS.from_product_state([shfs] * norb, product_states[0])

    if basis_state:
        # compute swap factor
        swap_factor = _compute_swap_factor(mps_reference)

        # apply swap factor
        if swap_factor == -1:
            minus_identity_npc = npc.Array.from_ndarray(
                -shfs.get_op("Id").to_ndarray(),
                [shfs.leg, shfs.leg.conj()],
                labels=["p", "p*"],
            )
            mps_reference.apply_local_op(0, minus_identity_npc)

        mps = mps_reference
    else:
        # initialize the TeNPy ExactDiag class instance
        charge_sector = mps_reference.get_total_charge(True)
        exact_diag = ExactDiag(
            FermiHubbardChain({"L": norb}), charge_sector=charge_sector
        )
        statevector_reference = exact_diag.mps_to_full(mps_reference)
        leg_charge = statevector_reference.legs[0]

        # determine the mapping from ffsim basis to TeNPy basis
        basis_ordering_ffsim, swap_factors_ffsim = _map_tenpy_to_ffsim_basis(
            product_states, exact_diag
        )
        basis_ordering_tenpy = np.argsort(basis_ordering_ffsim)
        swap_factors_tenpy = swap_factors_ffsim[np.argsort(basis_ordering_ffsim)]

        # convert ffsim statevector to TeNPy MPS
        statevector = np.multiply(swap_factors_tenpy, statevector[basis_ordering_tenpy])
        statevector_npc = npc.Array.from_ndarray(statevector, [leg_charge])
        mps = exact_diag.full_to_mps(statevector_npc)

    return mps


def _bitstring_to_product_state(bitstring: tuple[int, int], norb: int) -> list[int]:
    r"""Return the product state in TeNPy SpinHalfFermionSite notation.

    Args:
        bitstring: The bitstring in the form `(int_a, int_b)`.
        norb: The number of spatial orbitals.

    Returns:
        The product state in TeNPy SpinHalfFermionSite notation.
    """

    # unpack bitstrings
    int_a, int_b = bitstring
    string_a = format(int_a, f"0{norb}b")
    string_b = format(int_b, f"0{norb}b")

    # relabel using TeNPy SpinHalfFermionSite notation
    product_state = []
    for i, site in enumerate(zip(reversed(string_b), reversed(string_a))):
        site_occupation = int("".join(site), base=2)
        product_state.append(site_occupation)

    return product_state


def _generate_product_states(norb: int, nelec: tuple[int, int]) -> list[list[int]]:
    r"""Return the ffsim-ordered list of product states in TeNPy SpinHalfFermionSite
    notation.

    Args:
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.

    Returns:
        The ffsim-ordered list of product states in TeNPy SpinHalfFermionSite notation.
    """

    # generate the strings
    dim = ffsim.dim(norb, nelec)
    strings = ffsim.addresses_to_strings(
        range(dim), norb=norb, nelec=nelec, bitstring_type=ffsim.BitstringType.STRING
    )
    string_tuples = [
        (
            int(string[len(string) // 2 :], base=2),
            int(string[: len(string) // 2], base=2),
        )
        for string in strings
    ]

    # convert strings to product states
    product_states = []
    for bitstring in string_tuples:
        product_states.append(_bitstring_to_product_state(bitstring, norb))

    return product_states


def _compute_swap_factor(mps: MPS) -> int:
    r"""Compute the swap factor due to the conversion from TeNPy to ffsim bases.

    Args:
        mps: The MPS.

    Returns:
        The swap factor (+1 or -1).
    """

    norb = mps.L
    fs = FermionSite(conserve="N")
    alpha_sector = mps.expectation_value("Nu")
    beta_sector = mps.expectation_value("Nd")
    product_state_fs_tenpy = [
        int(val) for pair in zip(alpha_sector, beta_sector) for val in pair
    ]
    mps_fs = MPS.from_product_state([fs] * 2 * norb, product_state_fs_tenpy)

    tenpy_ordering = list(range(2 * norb))
    midpoint = len(tenpy_ordering) // 2
    mask1 = tenpy_ordering[:midpoint][::-1]
    mask2 = tenpy_ordering[midpoint:][::-1]
    ffsim_ordering = [int(val) for pair in zip(mask1, mask2) for val in pair]

    mps_ref = deepcopy(mps_fs)
    mps_ref.permute_sites(ffsim_ordering, swap_op=None)
    mps_fs.permute_sites(ffsim_ordering, swap_op="auto")
    swap_factor = cast(int, round(mps_fs.overlap(mps_ref)))

    return swap_factor


def _map_tenpy_to_ffsim_basis(
    product_states: list[list[int]], exact_diag: ExactDiag
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    r"""Return the mapping from the TeNPy basis to the ffsim basis.

    Args:
        product_states: The ffsim-ordered list of product states in TeNPy
            SpinHalfFermionSite notation.
        exact_diag: The exact diagonalization class instance.

    Returns:
        basis_ordering_ffsim: The permutation to map from the TeNPy to ffsim basis.
        swap_factors: The minus signs that are introduced due to this mapping.
    """

    basis_ordering_ffsim, swap_factors = [], []
    for i, state in enumerate(product_states):
        # basis_ordering_ffsim
        prod_mps = MPS.from_product_state(exact_diag.model.lat.mps_sites(), state)
        prod_statevector = list(exact_diag.mps_to_full(prod_mps).to_ndarray())
        idx = prod_statevector.index(1)
        basis_ordering_ffsim.append(idx)

        # swap_factors
        swap_factors.append(_compute_swap_factor(prod_mps))

    return np.array(basis_ordering_ffsim), np.array(swap_factors)
