from collections.abc import Iterator, Mapping

import numpy as np

class FermionOperator(Mapping[tuple[tuple[bool, bool, int], ...], complex]):
    """A fermionic operator.

    A FermionOperator represents a linear combination of products of fermionic creation
    and annihilation operators.
    """

    def __init__(
        self, coeffs: dict[tuple[tuple[bool, bool, int], ...], complex]
    ) -> None:
        """Initialize a FermionOperator.

        Args:
            coeffs: The coefficients of the operator.
        """
    def normal_ordered(self) -> "FermionOperator":
        """Return the normal ordered form of the operator.

        The normal ordered form of an operator is an equivalent operator in which
        each term has been reordered into a canonical ordering.
        In each term of a normal-ordered fermion operator, the operators comprising
        the term appear from left to right in descending lexicographic order by
        (action, spin, orb). That is, all creation operators appear before all
        annihilation operators; within creation/annihilation operators, spin beta
        operators appear before spin alpha operators, and larger orbital indices
        appear before smaller orbital indices.
        """
    def conserves_particle_number(self) -> bool:
        """Return whether the operator conserves particle number."""
    def conserves_spin_z(self) -> bool:
        """Return whether the operator conserves the Z component of spin."""
    def many_body_order(self) -> int:
        """Return the many-body order of the operator.

        The many-body order is defined as the length of the longest term contained
        in the operator.
        """
    def __getitem__(self, key: tuple[tuple[bool, bool, int], ...]) -> complex: ...
    def __contains__(self, key: object) -> bool: ...
    def __iter__(self) -> Iterator[tuple[tuple[bool, bool, int], ...]]: ...
    def __len__(self) -> int: ...

def apply_phase_shift_in_place(
    vec: np.ndarray, phase: complex, indices: np.ndarray
) -> None: ...
def apply_givens_rotation_in_place(
    vec: np.ndarray, c: float, s: complex, slice1: np.ndarray, slice2: np.ndarray
) -> None: ...
def gen_orbital_rotation_index_in_place(
    norb: int,
    nocc: int,
    linkstr_index: np.ndarray,
    diag_strings: np.ndarray,
    off_diag_strings: np.ndarray,
    off_diag_strings_index: np.ndarray,
    off_diag_index: np.ndarray,
) -> None: ...
def apply_single_column_transformation_in_place(
    vec: np.ndarray,
    column: np.ndarray,
    diag_val: complex,
    diag_strings: np.ndarray,
    off_diag_strings: np.ndarray,
    off_diag_index: np.ndarray,
) -> None: ...
def apply_num_op_sum_evolution_in_place(
    vec: np.ndarray,
    phases: np.ndarray,
    occupations: np.ndarray,
) -> None: ...
def apply_diag_coulomb_evolution_in_place_num_rep(
    vec: np.ndarray,
    mat_exp: np.ndarray,
    norb: int,
    mat_alpha_beta_exp: np.ndarray,
    occupations_a: np.ndarray,
    occupations_b: np.ndarray,
) -> None: ...
def apply_diag_coulomb_evolution_in_place_z_rep(
    vec: np.ndarray,
    mat_exp: np.ndarray,
    mat_exp_conj: np.ndarray,
    norb: int,
    mat_alpha_beta_exp: np.ndarray,
    mat_alpha_beta_exp_conj: np.ndarray,
    strings_a: np.ndarray,
    strings_b: np.ndarray,
) -> None: ...
def contract_diag_coulomb_into_buffer_num_rep(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    mat_alpha_beta: np.ndarray,
    occupations_a: np.ndarray,
    occupations_b: np.ndarray,
    out: np.ndarray,
) -> None: ...
def contract_diag_coulomb_into_buffer_z_rep(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    mat_alpha_beta: np.ndarray,
    strings_a: np.ndarray,
    strings_b: np.ndarray,
    out: np.ndarray,
) -> None: ...
def contract_num_op_sum_spin_into_buffer(
    vec: np.ndarray,
    coeffs: np.ndarray,
    occupations: np.ndarray,
    out: np.ndarray,
) -> None: ...
