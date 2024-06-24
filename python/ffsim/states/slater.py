import typing

import numpy as np
import scipy.linalg


def sample_slater(
    rdm: np.ndarray | tuple[np.ndarray, np.ndarray],
    chain_length: int,
    n_chains: int,
    n_particles_to_move: int = 1,
) -> np.ndarray:
    """
    Collect samples of electronic configurations from a Slater determinant defined
    by its one-body reduced density matrix (RDM). The sampler uses the
    Metropolis-Hastings method.

    Args:
        rdm: The one-body reduced density matrix that defines the Slater determinant
            This is either a single Numpy array specifying the 1-RDM of a
            spin-polarized system, or a pair of Numpy arrays where each element
            of the pair contains the 1-RDM for each spin sector.
        chain_length: Length of each sampling chain.
        n_chains: Number of chains to run. The total number of samples is given
            by (``chain_length`` + 1) * ``n_chains``.
        n_particles_to_move: At each step of the chain, the maximum number of
            electrons that are moved into a different orbital.
        seed: Seed for the random number generator.

    Returns:
        A 2D Numpy array with samples of electronic configurations.
        Each row is a sample.
    """

    if isinstance(rdm, np.ndarray) and rdm.ndim == 2:
        # spinless case
        typing.cast(np.ndarray, rdm)
        sampled_configuration = _sample_spinless(
            rdm, chain_length, n_chains, n_particles_to_move
        )
    else:
        # Spinful case
        rdm_a, rdm_b = rdm
        typing.cast(np.ndarray, rdm_a)
        typing.cast(np.ndarray, rdm_b)

        sampled_configuration_a = _sample_spinless(
            rdm_a, chain_length, n_chains, n_particles_to_move
        )

        sampled_configuration_b = _sample_spinless(
            rdm_b, chain_length, n_chains, n_particles_to_move
        )

        sampled_configuration = np.concatenate(
            (sampled_configuration_a, sampled_configuration_b), axis=1
        )

    return sampled_configuration


def _propose_move(
    previous_step: np.ndarray, norb: int, n_particles_to_move: int
) -> np.ndarray:
    """
    Proposes a new state in the Markov chain. The new state is obtained by allowing
    at most ``n_particles_to_move`` to change the orbital that they occupy.
    The new orbitals that they can occupy are chosen at random from the set of
    unoccupied orbitals.

    Args:
        previous step: A Numpy array with the starting positions of the electrons.
        norb: Number of orbitals.
        n_particles_to_move: At each step of the chain, the maximum number of
            electrons that are moved into a different orbital.

    Returns:
        A Numpy array with the proposed new set of positions of the electrons.
    """

    n_chains = previous_step.shape[0]
    nelec = previous_step.shape[1]

    positions = np.arange(norb)

    new_state = np.zeros((n_chains, nelec), dtype=int)

    for i in range(n_chains):
        sample_particles_to_move = np.random.choice(
            np.arange(n_particles_to_move) + 1, 1
        )

        open_orbitals = np.setdiff1d(
            positions, previous_step[i], assume_unique=True)

        electron_id = np.sort(
            np.random.choice(np.arange(nelec),
                             sample_particles_to_move, replace=False)
        )

        new_positions = np.random.choice(
            open_orbitals, sample_particles_to_move, replace=False
        )

        new_state[i] = previous_step[i]
        new_state[i, electron_id] = new_positions
        new_state[i] = np.sort(new_state[i])

    return new_state


def _evaluate_logdeterminant_squared(
    positions: np.ndarray, phi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a batch of positions of electronic configurations, evaluate their
    Slater determinant.

    Args:
        positions: A 2D Numpy array of electronic configurations. Each row represents
            the position of the electrons for each configuration.
        phi: A 2D (norb, nelec) Numpy array. The matrix must be column-wise
            orthogonal. Each column represents an orbital.

    Returns: Tuple. The first element is a Numpy array with the signs of the
        determinants. The second element is a Numpy array with the logarithm
        of the absolute value of the determinant.
    """

    det_matrices = phi[positions, :]

    signs, logdet_amps = np.linalg.slogdet(det_matrices)

    return signs, logdet_amps


def _accept(
    logdet_squared_new: tuple[np.ndarray, np.ndarray],
    logdet_squared_old: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    Metropolis-Hastings acceptance of the new state of the chain.

    Args:
        logdet_squared_new: Tuple. The first element of the tuple is a Numpy
            array with the sign of the determinant for the proposed electronic
            configurations for each chain. The second element of the tuple is a
            Numpy array with the logarithm of the absolute value of the
            determinant of the proposed configurations.
        logdet_squared_old: Tuple. The first element of the tuple is a Numpy
            array with the sign of the determinant for the previous electronic
            configurations for each chain. The second element of the tuple is a
            Numpy array with the logarithm of the absolute value of the
            determinant of the previous configurations.

    Returns:
        A Numpy array of ``bool`` type that specifies if the move for each chain
        is accepted (``True``) of rejected (``False``).
    """

    n_chains = len(logdet_squared_new[0])
    random_probs = np.random.rand(n_chains)
    ones_vector = np.ones(n_chains)

    p_acceptance = np.minimum(
        ones_vector,
        np.abs(logdet_squared_new[0])
        * np.exp(2 * (logdet_squared_new[1] - logdet_squared_old[1])),
    )

    return p_acceptance >= random_probs


def _initialize_chains(
    n_chains: int, nelec: int, norb: int, rdm: np.ndarray
) -> np.ndarray:
    """
    Proposes initial positions for the electrons to initialize the Markov chains.
    Electron positions are sampled from the diagonal part of the one-body
    RDM.

    Args:
        n_chains: Number of chains.
        nelec: Number of electrons.
        norb: Number of orbitals.
        rdm: A Numoy array with the one-body RDM.

    Returns:
        A 2D Numpy array with the initial electronic configuration for each
        chain. Each row is the initial electronic configuration for each chain.
    """

    positions = np.zeros((n_chains, nelec))
    count = 0
    while count < n_chains:
        attempt = np.sort(
            np.random.choice(
                np.arange(norb), size=nelec, replace=True, p=np.diag(rdm) / nelec
            )
        )

        if len(np.unique(attempt)) == nelec:
            positions[count] = attempt
            count += 1

    return positions.astype("int")


def _select(
    acceptance_vector: np.ndarray, new_pos: np.ndarray, old_pos: np.ndarray
) -> np.ndarray:
    """
    Based on the acceptance criterion of the Markov chain returns the state for
    the next step of the chain.

    Args:
        acceptance_vector: A 1D Numpy array of ``bool`` type. ``True``/ ``False``
            entries represent accepted or rejected moves respectively.
        new_pos: A 2D Numpy array with the proposed new positions of the
            electrons. Each row represents an electronic configuration.
        old_pos: A 2D Numpy array with the previous positions of the
            electrons. Each row represents an electronic configuration.

    Returns:
        A 2D Numpy array with the previous positions of the
        electrons after accepting or rejecting the electron moves according to
        the ``acceptance_vector``.


    """
    for i in range(new_pos.shape[0]):
        if not acceptance_vector[i]:
            new_pos[i] = old_pos[i]

    return new_pos


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


def _sample_spinless(
    rdm: np.ndarray,
    chain_length: int,
    n_chains: int,
    n_particles_to_move: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Collect samples of electronic configurations from a Slater determinant defined
    by its one-body reduced density matrix (RDM). The sampler uses the
    Metropolis-Hastings method.

    Args:
        rdm: The one-body reduced density matrix that defines the Slater determinant
            This is a single Numpy array specifying the 1-RDM of a spin-polarized
            system.
        chain_length: Length of each sampling chain.
        n_chains: Number of chains to run. The total number of samples is given
            by (``chain_length`` + 1) * ``n_chains``.
        n_particles_to_move: At each step of the chain, the maximum number of
            electrons that are moved into a different orbital.
        seed: Seed for the random number generator.

    Returns:
        A 2Dimensional Numpy array with samples of electronic configurations.
        Each row is a sample.
    """

    np.random.seed(seed)

    norb = rdm.shape[0]

    nelec = round(np.sum(np.diag(rdm)))

    if nelec < n_particles_to_move:
        raise ValueError(
            "Number of electrons smaller than ``n_particles_to_move``: "
            f"number of electrons ({nelec}) "
            f"and ``n_particles_to_move`` ({n_particles_to_move})."
        )

    _, vecs = scipy.linalg.eigh(rdm)

    phi = vecs[:, -nelec:]

    positions = np.zeros((n_chains, chain_length + 1, nelec), dtype=int)

    positions[:, 0, :] = _initialize_chains(n_chains, nelec, norb, rdm)

    for i in range(chain_length):
        proposed_move = _propose_move(
            positions[:, i, :], norb, n_particles_to_move)

        slogdet_new = _evaluate_logdeterminant_squared(proposed_move, phi)
        slogdet_old = _evaluate_logdeterminant_squared(positions[:, i, :], phi)

        accept_move = _accept(slogdet_new, slogdet_old)

        positions[:, i + 1,
                  :] = _select(accept_move, proposed_move, positions[:, i, :])

    return _positions_to_fock(
        np.reshape(positions, ((chain_length + 1) * n_chains, nelec)), norb
    )
