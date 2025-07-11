import ffsim.variational
import pyscf
import ffsim

import numpy as np
from opt_einsum import contract
import scipy
import jax
import jax.numpy as jnp
from jax import grad

def _df_tensors_to_params(
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    diag_coulomb_mat_mask: np.ndarray,
):
    _, norb, _ = orbital_rotations.shape
    leaf_logs = [scipy.linalg.logm(mat) for mat in orbital_rotations]
    # include the diagonal element
    # TODO this discards the imaginary part of the logarithm, see if we can do better
    leaf_param_real_indices = np.triu_indices(norb, k=1)
    leaf_params_real = np.real(
        np.ravel(
            np.array([leaf_log[leaf_param_real_indices] for leaf_log in leaf_logs])
        )
    )
    # add imag part
    leaf_param_imag_indices = np.triu_indices(norb)
    leaf_params_imag = np.imag(
        np.ravel(
            np.array([leaf_log[leaf_param_imag_indices] for leaf_log in leaf_logs])
        )
    )
    core_param_indices = np.nonzero(diag_coulomb_mat_mask)
    core_params = np.ravel(
        np.array(
            [
                diag_coulomb_mat[core_param_indices]
                for diag_coulomb_mat in diag_coulomb_mats
            ]
        )
    )
    # print("leaf_params_real[:2]")
    # print(leaf_params_real[:2])
    # print("leaf_params_imag[:2]")
    # print(leaf_params_imag[:2])
    # print("len(leaf_params_real)")
    # print(len(leaf_params_real))
    x = jnp.concatenate([leaf_params_real, leaf_params_imag, core_params])
    return x


def _params_to_leaf_logs(params: jnp.ndarray, n_tensors: int, norb: int):
    leaf_imag_logs = jnp.zeros((n_tensors, norb, norb), dtype="complex")
    leaf_logs = jnp.zeros((n_tensors, norb, norb), dtype="complex")
    # reconstruct the real part
    triu_indices = jnp.triu_indices(norb, k=1)
    param_length = len(triu_indices[0])
    for i in range(n_tensors):
        leaf_logs = (
            leaf_logs
            .at[i, triu_indices[0], triu_indices[1]]
            .set(params[i * param_length : (i + 1) * param_length])
        )
        leaf_logs[i] -= leaf_logs[i].T
    # reconstruct the imag part
    triu_indices = jnp.triu_indices(norb)
    real_begin_index = param_length * n_tensors
    param_length = len(triu_indices[0])
    for i in range(n_tensors):
        leaf_imag_logs[i][triu_indices] += (
            1j
            * params[
                i * param_length + real_begin_index : (i + 1) * param_length
                + real_begin_index
            ]
        )
        leaf_imag_logs_transpose = leaf_imag_logs[i].T
        # keep the diagonal element
        diagonal_element = jnp.diag(jnp.diag(leaf_imag_logs_transpose))
        leaf_imag_logs[i] += leaf_imag_logs_transpose
        leaf_imag_logs[i] -= diagonal_element
    leaf_logs += leaf_imag_logs
    return leaf_logs


def _expm_antihermitian(mats: jnp.ndarray) -> jnp.ndarray:
    eigs, vecs = jnp.linalg.eigh(-1j * mats)
    return jnp.einsum("tij,tj,tkj->tik", vecs, jnp.exp(1j * eigs), vecs.conj())


def _params_to_df_tensors(
    params: jnp.ndarray, n_tensors: int, norb: int, diag_coulomb_mat_mask: jnp.ndarray
):
    leaf_logs = _params_to_leaf_logs(params, n_tensors, norb)
    orbital_rotations = _expm_antihermitian(leaf_logs)
    # orbital_rotations = scipy.linalg.expm(leaf_logs)
    n_leaf_params = n_tensors * (norb * (norb - 1) // 2 + norb * (norb + 1) // 2)
    core_params = jnp.real(params[n_leaf_params:])
    param_indices = jnp.nonzero(diag_coulomb_mat_mask)
    param_length = len(param_indices[0])
    diag_coulomb_mats = jnp.zeros((n_tensors, norb, norb))
    for i in range(n_tensors):
        diag_coulomb_mats[i][param_indices] = core_params[
            i * param_length : (i + 1) * param_length
        ]
        diag_coulomb_mats[i] += diag_coulomb_mats[i].T
        diag_coulomb_mats[i][range(norb), range(norb)] /= 2
    return diag_coulomb_mats, orbital_rotations


def double_factorized_t2_compress(
    t2: jnp.ndarray,
    diag_coulomb_mats: jnp.ndarray,
    orbital_rotations: jnp.ndarray,
    *,
    tol: float = 1e-8,
    n_reps: int | None = None,
    interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]
    | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    _, _, norb, _ = orbital_rotations.shape
    diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)[:n_reps]
    # not dealing with stack for now, see line 185
    # diag_coulomb_mats = jnp.stack([diag_coulomb_mats, diag_coulomb_mats], axis=1)
    orbital_rotations = orbital_rotations.reshape(-1, norb, norb)[:n_reps]
    n_tensors, norb, _ = orbital_rotations.shape

    def fun(x):
        diag_coulomb_mats, orbital_rotations = _params_to_df_tensors(
            x, n_tensors, norb, diag_coulomb_mask
        )
        reconstructed = (
            1j
            * contract(
                "mpq,map,mip,mbq,mjq->ijab",
                diag_coulomb_mats,
                orbital_rotations,
                orbital_rotations.conj(),
                orbital_rotations,
                orbital_rotations.conj(),
                # optimize="greedy"
            )[:nocc, :nocc, nocc:, nocc:]
        )
        diff = reconstructed - t2
        # print(reconstructed)

        return 0.5 * jnp.sum(jnp.abs(diff) ** 2)

    def callback(intermediate_result: scipy.optimize.OptimizeResult):
        print(f"Intermediate result: loss {intermediate_result.fun:.8f}")

    method = "L-BFGS-B"
    # method = "trust-constr"
    # method = "COBYQA"
    # method = "COBYLA"
    options = {"maxiter": 100}
    
    pairs_aa, pairs_ab = interaction_pairs

    # Zero out diagonal coulomb matrix entries
    pairs = []
    if pairs_aa is not None:
        pairs += pairs_aa
    if pairs_ab is not None:
        pairs += pairs_ab
    if not pairs:
        diag_coulomb_mask = np.ones((norb, norb), dtype=bool)
    else:
        diag_coulomb_mask = np.zeros((norb, norb), dtype=bool)
        rows, cols = zip(*pairs)
        diag_coulomb_mask[rows, cols] = True
        diag_coulomb_mask[cols, rows] = True

    diag_coulomb_mask = np.triu(diag_coulomb_mask)
    
    print(orbital_rotations.shape)

    x0 = _df_tensors_to_params(diag_coulomb_mats, orbital_rotations, diag_coulomb_mask)
    grad(fun(x0))
    # print(f"initial val: {fun(x0)}")
    # print(orbital_rotations[0])
    # diag_coulomb_mats_converted, orbital_rotations_converted = _params_to_df_tensors(
    #     x0, n_tensors, norb, diag_coulomb_mask
    # )
    result = scipy.optimize.minimize(
        fun,
        x0,
        method=method,
        jac=False,
        callback=callback,
        options=options,
        # fun, x0, method=method, jac=True, callback=callback, options=options
        # fun, x0, method=method
    )
    # print(all(result.x == x0))
    print(fun(result.x))
    diag_coulomb_mats, orbital_rotations = _params_to_df_tensors(
        result.x, n_tensors, norb, diag_coulomb_mask
    )
    # stack here
    diag_coulomb_mats = jnp.stack([diag_coulomb_mats, diag_coulomb_mats], axis=1)
    return diag_coulomb_mats, orbital_rotations


def from_t_amplitudes_compressed(
    t2: jnp.ndarray,
    *,
    t1: jnp.ndarray | None = None,
    n_reps: int | None = None,
    interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]
    | None = None,
    tol: float = 1e-8,
    optimize=False,
) -> ffsim.UCJOpSpinBalanced:
    if interaction_pairs is None:
        interaction_pairs = (None, None)
    pairs_aa, pairs_ab = interaction_pairs

    nocc, _, nvrt, _ = t2.shape
    norb = nocc + nvrt

    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2(
        t2, tol=tol
    )
    if optimize:
        diag_coulomb_mats, orbital_rotations = double_factorized_t2_compress(
            t2,
            diag_coulomb_mats,
            orbital_rotations,
            n_reps=n_reps,
            interaction_pairs=interaction_pairs,
        )
    else:
        diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)[:n_reps]
        diag_coulomb_mats = jnp.stack([diag_coulomb_mats, diag_coulomb_mats], axis=1)
        orbital_rotations = orbital_rotations.reshape(-1, norb, norb)[:n_reps]

        n_vecs, _, _, _ = diag_coulomb_mats.shape
        if n_reps is not None and n_vecs < n_reps:
            # Pad with no-ops to the requested number of repetitions
            diag_coulomb_mats = jnp.concatenate(
                [diag_coulomb_mats, jnp.zeros((n_reps - n_vecs, 2, norb, norb))]
            )
            eye = jnp.eye(norb)
            orbital_rotations = jnp.concatenate(
                [orbital_rotations, jnp.stack([eye for _ in range(n_reps - n_vecs)])]
            )

        # Zero out diagonal coulomb matrix entries if requested
        if pairs_aa is not None:
            mask = jnp.zeros((norb, norb), dtype=bool)
            rows, cols = zip(*pairs_aa)
            mask[rows, cols] = True
            mask[cols, rows] = True
            diag_coulomb_mats[:, 0] *= mask
        if pairs_ab is not None:
            mask = jnp.zeros((norb, norb), dtype=bool)
            rows, cols = zip(*pairs_ab)
            mask[rows, cols] = True
            mask[cols, rows] = True
            diag_coulomb_mats[:, 1] *= mask

    return None
    # final_orbital_rotation = None
    # if t1 is not None:
    #     final_orbital_rotation = orbital_rotation_from_t1_amplitudes(t1)

    # return ffsim.UCJOpSpinBalanced(
    #     diag_coulomb_mats=diag_coulomb_mats,
    #     orbital_rotations=orbital_rotations,
    #     final_orbital_rotation=final_orbital_rotation,
    # )


# Build N2 molecule
mol = pyscf.gto.Mole()
mol.build(
    atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
    basis="sto-6g",
    # basis="6-31g",
    symmetry="Dooh",
)

# Define active space
n_frozen = 2
active_space = range(n_frozen, mol.nao_nr())

# Get molecular data and Hamiltonian
scf = pyscf.scf.RHF(mol).run()
mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
norb, nelec = mol_data.norb, mol_data.nelec
mol_hamiltonian = mol_data.hamiltonian
print(f"norb = {norb}")
print(f"nelec = {nelec}")

# Get CCSD t2 amplitudes for initializing the ansatz
ccsd = pyscf.cc.CCSD(
    scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
).run()

nocc, _, nvrt, _ = ccsd.t2.shape

# Use 2 ansatz layers
n_reps = 2

# Use interactions implementable on a square lattice
pairs_aa = [(p, p + 1) for p in range(norb - 1)]
pairs_ab = [(p, p) for p in range(norb)]

ucj_op = from_t_amplitudes_compressed(
    ccsd.t2,
    t1=ccsd.t1,
    n_reps=n_reps,
    interaction_pairs=(pairs_aa, pairs_ab),
    optimize=True,
)


# if we drop imag
# converged SCF energy = -108.464957764796
# norb = 8
# nelec = (5, 5)
# E(CCSD) = -108.5933309085008  E_corr = -0.1283731437052353
# initial val: 0.027038428162185598
# False
# True
# 0.027038428162185598
