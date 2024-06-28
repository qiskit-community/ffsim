from __future__ import annotations

import numpy as np


def sample_slater(
    rdm: np.ndarray | tuple[np.ndarray, np.ndarray],
    shots: int,
    seed: np.random.Generator | int | None = None,
) -> np.ndarray:
    """
    Collect samples of electronic configurations from a Slater determinant.

    The Slater determinat is defined by its one-body reduced density matrix (RDM).
    The sampler uses a determinantal point process to auto-regressively produce
    uncorrelated samples.

    Args:
        rdm: The one-body reduced density matrix that defines the Slater determinant
            This is either a single Numpy array specifying the 1-RDM of a
            spin-polarized system, or a pair of Numpy arrays where each element
            of the pair contains the 1-RDM for each spin sector.
        shots: Number of samples to collect.
        seed: Either a Numpy random generator, an integer seed for the random number
            generator or ``None``.

    Returns:
        A 2D Numpy array with samples of electronic configurations.
        Each row is a sample.
    """

    rng = np.random.default_rng(seed)

    if isinstance(rdm, np.ndarray) and rdm.ndim == 2:
        # spinless case
        n = round(np.real(np.sum(np.diag(rdm))))
        norb, _ = rdm.shape

        if n == 0 or n == norb:
            sampled_configuration = np.ones((shots, norb), dtype=int) * int(n / norb)
        else:
            sampled_configuration = _sample_spinless_direct(rdm, shots, rng)
    else:
        # Spinful case
        rdm_a, rdm_b = rdm
        n_a = round(np.real(np.sum(np.diag(rdm_a))))
        n_b = round(np.real(np.sum(np.diag(rdm_b))))
        norb = rdm_a.shape[0]

        if n_a == 0 or n_a == norb:
            sampled_configuration_a = np.ones((shots, norb), dtype=int) * int(
                n_a / norb
            )
        else:
            sampled_configuration_a = _sample_spinless_direct(rdm_a, shots, rng)

        if n_b == 0 or n_b == norb:
            sampled_configuration_b = np.ones((shots, norb), dtype=int) * int(
                n_a / norb
            )
        else:
            sampled_configuration_b = _sample_spinless_direct(rdm_b, shots, rng)

        sampled_configuration = np.concatenate(
            (sampled_configuration_b, sampled_configuration_a), axis=1
        )

    return sampled_configuration


def _generate_conditionals_unormalized(
    rdm: np.ndarray, pos_array: np.ndarray, empty_orbitals: np.ndarray, marginal: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates the conditional and marginal probabilities for adding a particle
    to the available empty orbitals.

    This is a step of the autoregressive sampling, and uses Bayes's rule.

    Args:
        rdm: A Numpy array with the one-body reduced density matrix.
        pos_array: A numpy array with the positions of the particles.
        empty_orbitals: A numpy array with the empty orbitals that a new particle
            may occupy. The sorted union of ``pos_array`` and ``empty_orbitals``
            must be equal to ``numpy.arange(num_orbitals)``.
        marginal: The marginal probability associated to having the particles
            in the positions determined by ``pos_array``.
    Returns:
        A tuple of Numpy arrays. The first is the unormalized conditionals for
        adding a particle to any of the empty orbitals specified in the input.
        The second is the marginal corresponding to having the particles in the
        position array and one extra in all possible empty orbitals. Both arrays
        follow the same order as ``empty_orbitals``.

    """

    conditionals = np.zeros(len(empty_orbitals), dtype=complex)
    marginals = np.zeros(len(empty_orbitals), dtype=complex)

    for i, orbital in enumerate(empty_orbitals):
        new_pos_array = np.append(pos_array, [orbital])
        rest_rdm = rdm[new_pos_array, :]
        rest_rdm = rest_rdm[:, new_pos_array]
        marginals[i] = np.linalg.det(rest_rdm)
        conditionals[i] = marginals[i] / marginal

    return conditionals, marginals


def _autoregressive_slater(
    rdm: np.ndarray,
    norb: int,
    nelec: int,
    seed: np.random.Generator | int | None = None,
) -> np.ndarray:
    """
    Autoregressively sample positions of particles for a Slater-determinant wave
    function using a determinantal point process.

    Args:
        rdm: A numpy array with the one-body reduced density matrix.
        norb: Number of orbitals.
        nelec: Number of electrons.
        seed: Either a Numpy random generator, an integer seed for the random number
            generator or ``None``.
    Returns:
        A numpy array with the position of the sampled electrons.
    """
    rng = np.random.default_rng(seed)

    position = []
    marginal = []
    position.append(
        rng.choice(np.arange(norb), size=1, p=np.real(np.diag(rdm) / nelec))[0]
    )
    marginal.append(np.real(np.diag(rdm) / nelec)[position[0]])

    for k in range(nelec - 1):
        pos_array = np.array(position, dtype=int)

        empty_orbitals = np.setdiff1d(np.arange(norb), pos_array, assume_unique=True)

        u_conditionals, marginals = _generate_conditionals_unormalized(
            rdm, pos_array, empty_orbitals, marginal[-1]
        )

        conditionals = np.real(u_conditionals)
        conditionals /= np.sum(conditionals)

        n_empty_orbitals = len(empty_orbitals)

        index = rng.choice(np.arange(n_empty_orbitals), size=1, p=conditionals)[0]
        new_pos = empty_orbitals[index]

        position.append(new_pos)
        position = list(np.sort(np.array(position)))
        marginal.append(marginals[index])
    return np.array(position, dtype=int)


def _positions_to_fock(positions: np.ndarray, norb: int) -> np.ndarray:
    """
    Transforms electronic configurations defined by the position of the electrons
    to the occupation representation.

    Args:
        positions: A Numpy array of positions of electrons. Each row corresponds
            to an electronic configuration.
        norb: number of orbitals.

    Returns:
        A 2Dimensional Numpy array of electronic configurations represented by
        the orbital occupancy. Each row is an electronic configuration.
    """

    n_samples = positions.shape[0]
    fock = np.zeros((n_samples, norb), dtype=int)
    for i in range(n_samples):
        fock[i, norb - 1 - positions[i]] = 1
    return fock


def _sample_spinless_direct(
    rdm: np.ndarray,
    shots: int,
    seed: np.random.Generator | int | None = None,
) -> np.ndarray:
    """
    Collect samples of electronic configurations from a Slater determinant for
    spin-polarized systems.

    The Slater determinat is defined by its one-body reduced density matrix (RDM).
    The sampler uses a determinantal point process to auto-regressively produce
    uncorrelated samples.

    Args:
        rdm: The one-body reduced density matrix that defines the Slater determinant
            This is either a single Numpy array specifying the 1-RDM of a
            spin-polarized system, or a pair of Numpy arrays where each element
            of the pair contains the 1-RDM for each spin sector.
        shots: Number of samples to collect.
        seed: Either a Numpy random generator, an integer seed for the random number
            generator or ``None``.

    Returns:
        A 2D Numpy array with samples of electronic configurations.
        Each row is a sample.
    """

    rng = np.random.default_rng(seed)
    norb, _ = rdm.shape
    nelec = round(np.real(np.sum(np.diag(rdm))))
    positions = np.zeros((shots, nelec), dtype=int)

    for i in range(shots):
        positions[i] = _autoregressive_slater(rdm, norb, nelec, rng)

    return _positions_to_fock(positions, norb)
