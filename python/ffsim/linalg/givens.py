# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utilities for performing the Givens decomposition."""

from __future__ import annotations

import cmath
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

from ffsim import _lib

jax.config.update("jax_enable_x64", True)


def apply_matrix_to_slices(
    target: np.ndarray,
    mat: np.ndarray,
    slices,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a matrix to slices of a target tensor.

    Args:
        target: The tensor containing the slices on which to apply the matrix.
        mat: The matrix to apply to slices of the target tensor.
        slices: The slices of the target tensor on which to apply the matrix.

    Returns:
        The resulting tensor.
    """
    if out is target:
        raise ValueError("Output buffer cannot be the same as the input")
    if out is None:
        out = np.empty_like(target)
    out[...] = target[...]
    for i, slice_i in enumerate(slices):
        out[slice_i] *= mat[i, i]
        for j, slice_j in enumerate(slices):
            if j != i:
                out[slice_i] += mat[i, j] * target[slice_j]
    return out


def givens_decomposition(
    mat: np.ndarray,
    tol: float = 1e-12,
    *,
    max_givens: int | None = None,
    max_layers: int | None = None,
    **optimize_kwargs,
) -> tuple[list[tuple[float, complex, int, int]], np.ndarray]:
    r"""Givens rotation decomposition of a unitary matrix.

    The Givens rotation decomposition of an :math:`n \times n` unitary matrix :math:`U`
    is given by

    .. math::

        U = D G_L^* G_{L-1}^* \cdots G_1^*

    where :math:`D` is a diagonal matrix and each :math:`G_k` is a Givens rotation.
    Here, the star :math:`*` denotes the element-wise complex conjugate.
    A Givens rotation acts on the two-dimensional subspace spanned by the :math:`i`-th
    and :math:`j`-th basis vectors as

    .. math::

        \begin{pmatrix}
            c & s \\
            -s^* & c \\
        \end{pmatrix}

    where :math:`c` is a real number and :math:`s` is a complex number.
    Therefore, a Givens rotation is described by a 4-tuple
    :math:`(c, s, i, j)`, where :math:`c` and :math:`s` are the numbers appearing
    in the rotation matrix, and :math:`i` and :math:`j` are the
    indices of the basis vectors of the subspace being rotated.
    This function always returns Givens rotations with the property that
    :math:`i` and :math:`j` differ by at most one, that is, either :math:`j = i + 1`
    or :math:`j = i - 1`.

    The number of Givens rotations :math:`L` is at most :math:`\frac{n (n-1)}{2}`,
    but it may be less. If we think of Givens rotations acting on disjoint indices
    as operations that can be performed in parallel, then the entire sequence of
    rotations can always be performed using at most :math:`n` layers of parallel
    operations. The decomposition algorithm is described in the reference below.

    **Compression.** By default this function returns an exact decomposition. If
    ``max_givens`` or ``max_layers`` is specified, the exact decomposition is
    *trimmed* to use at most that many Givens rotations or brickwork layers,
    respectively (never more than the exact decomposition already contains, so a
    near-identity matrix may use fewer rotations than the budget). The retained
    rotations lie at the beginning of the brickwork pattern, and their angles
    (together with the diagonal phases) are numerically optimized to minimize the
    Hilbert-Schmidt infidelity
    :math:`1 - \lvert \operatorname{Tr}(U^\dagger V) \rvert^2 / n^2` between the
    original matrix :math:`U` and the reconstructed matrix :math:`V`. Because this
    objective is invariant to a global phase, the global phase of the reconstructed
    matrix is ignored when the decomposition is compressed. The returned decomposition
    is then only approximate. When both ``max_givens`` and ``max_layers`` are given, the
    tighter of the two constraints is applied. Note that when the decomposition is
    compressed, ``tol`` is not respected: the optimized angles are chosen to best
    approximate :math:`U`, so the reconstructed matrix may differ from :math:`U` by
    more than ``tol``.

    References:
        - `Clements et al., "Optimal design for universal multiport interferometers" (2016)`_

    Args:
        mat: The unitary matrix to decompose into Givens rotations.
        tol: Matrix entries smaller than this value will be treated as equal to zero
            when computing the exact decomposition (which the compressed path also
            starts from).
        max_givens: The maximum number of Givens rotations to use. If specified, the
            decomposition is compressed to use at most this many Givens rotations.
        max_layers: The maximum number of brickwork layers to use. If specified, the
            decomposition is compressed to use at most this many layers. The full
            brickwork pattern has :math:`n` layers.
        optimize_kwargs: Keyword arguments to pass to :func:`scipy.optimize.minimize`,
            which performs the optimization when the decomposition is compressed.

    Returns:
        - A list containing the Givens rotations :math:`G_1, \ldots, G_L`.
          Each Givens rotation is represented as a 4-tuple
          :math:`(c, s, i, j)`, where :math:`c` and :math:`s` are the numbers appearing
          in the rotation matrix, and :math:`i` and :math:`j` are the
          indices of the basis vectors of the subspace being rotated.
        - A Numpy array containing the diagonal elements of the matrix :math:`D`.

    .. _Clements et al., "Optimal design for universal multiport interferometers" (2016): https://doi.org/10.1364/OPTICA.3.001460
    """  # noqa: E501
    mat = mat.astype(complex, copy=False)
    if max_givens is None and max_layers is None:
        return _lib.givens_decomposition(mat, tol=tol)
    return _givens_decomposition_compressed(
        mat,
        tol=tol,
        max_givens=max_givens,
        max_layers=max_layers,
        **optimize_kwargs,
    )


def givens_decomposition_slater(
    orbital_coeffs: np.ndarray,
    tol: float = 1e-12,
    *,
    max_givens: int | None = None,
    max_layers: int | None = None,
    **optimize_kwargs,
) -> list[tuple[float, complex, int, int]]:
    r"""Givens rotation decomposition for Slater determinant preparation.

    Given the coefficient matrix of the occupied orbitals of a Slater determinant,
    returns a sequence of Givens rotations that, when applied to the electronic
    configuration :math:`\lvert 1 \cdots 1 0 \cdots 0 \rangle` (with the first
    :math:`m` orbitals occupied), prepares the Slater determinant. Here
    ``orbital_coeffs`` is an :math:`m \times n` matrix whose rows are the :math:`m`
    occupied orbitals expressed in a basis of :math:`n` spatial orbitals; its rows are
    assumed to be orthonormal.

    Unlike :func:`givens_decomposition`, this decomposition is specialized for state
    preparation: it only needs to prepare the :math:`m` occupied orbitals rather than a
    full :math:`n \times n` orbital rotation, so it uses at most :math:`m (n - m)`
    Givens rotations arranged in a diamond-shaped pattern. The decomposition contains no
    diagonal phases, because a global phase and any rotation within the occupied space
    leave the prepared Slater determinant unchanged.

    A Givens rotation is described by a 4-tuple :math:`(c, s, i, j)`, where :math:`c` is
    a real number, :math:`s` is a complex number, and :math:`i` and :math:`j` are the
    (adjacent) indices of the orbitals being rotated.

    **Compression.** By default this function returns an exact decomposition. If
    ``max_givens`` or ``max_layers`` is specified, the exact decomposition is *trimmed*
    to use at most that many Givens rotations or brickwork layers, respectively (never
    more than the exact decomposition already contains, so a near-identity matrix may
    use fewer rotations than the budget). The retained rotations lie at the beginning
    of the pattern, and their angles are numerically optimized to maximize the fidelity
    :math:`\lvert \det(A B^\dagger) \rvert^2` between the prepared Slater determinant
    (occupied orbitals :math:`A`) and the target (occupied orbitals :math:`B`). The
    returned decomposition is then only approximate. When both ``max_givens`` and
    ``max_layers`` are given, the tighter of the two constraints is applied. Note that
    when the decomposition is compressed, ``tol`` is not respected: the optimized
    angles are chosen to best approximate the target Slater determinant, so the
    prepared state may differ from the target by more than ``tol``.

    Args:
        orbital_coeffs: The :math:`m \times n` matrix of occupied orbital coefficients.
        tol: Matrix entries smaller than this value will be treated as equal to zero
            when computing the exact decomposition (which the compressed path also
            starts from).
        max_givens: The maximum number of Givens rotations to use. If specified, the
            decomposition is compressed to use at most this many Givens rotations.
        max_layers: The maximum number of brickwork layers to use. If specified, the
            decomposition is compressed to use at most this many layers.
        optimize_kwargs: Keyword arguments to pass to :func:`scipy.optimize.minimize`,
            which performs the optimization when the decomposition is compressed.

    Returns:
        A list containing the Givens rotations, each represented as a 4-tuple
        :math:`(c, s, i, j)`.
    """
    orbital_coeffs = orbital_coeffs.astype(complex, copy=False)
    if max_givens is None and max_layers is None:
        return _lib.givens_decomposition_slater(orbital_coeffs, tol)
    return _givens_decomposition_slater_compressed(
        orbital_coeffs,
        tol=tol,
        max_givens=max_givens,
        max_layers=max_layers,
        **optimize_kwargs,
    )


def _greedy_layer_ids(interaction_pairs: list[tuple[int, int]]) -> list[int]:
    """Assign each Givens rotation to a parallel layer by greedy ASAP scheduling.

    The exact decomposition returns rotations in a valid application order. Each
    rotation is greedily placed in the earliest layer in which both of its orbitals
    are free, respecting the order of the rotations. Returns a parallel list giving
    the layer index of each rotation. Because ``_lib.givens_decomposition`` already
    returns rotations in a valid dependency order, this greedy schedule achieves the
    minimum number of layers, so a sparse decomposition maps to a shallow pattern.

    The rotation pairs are treated symmetrically (their order within a pair does not
    matter), so this works for both the orbital-rotation decomposition (where a pair
    may have ``i > j``) and the Slater decomposition.
    """
    last_layer: dict[int, int] = {}
    layer_ids = []
    for i, j in interaction_pairs:
        layer_id = max(last_layer.get(i, -1), last_layer.get(j, -1)) + 1
        layer_ids.append(layer_id)
        last_layer[i] = layer_id
        last_layer[j] = layer_id
    return layer_ids


def _validate_caps(max_givens: int | None, max_layers: int | None) -> None:
    """Validate that the compression caps are non-negative."""
    if max_givens is not None and max_givens < 0:
        raise ValueError(f"max_givens must be non-negative. Got {max_givens}.")
    if max_layers is not None and max_layers < 0:
        raise ValueError(f"max_layers must be non-negative. Got {max_layers}.")


def _rotations_to_angles(
    givens_rotations: list[tuple[float, complex, int, int]],
) -> tuple[list[tuple[int, int]], list[float], list[float]]:
    """Convert ``(c, s, i, j)`` rotations to ``(pairs, thetas, phis)``.

    Inverts the ``c = cos(theta)``, ``s = sin(theta) * exp(1j * phi)`` parametrization
    used by :func:`_angles_to_rotations`.
    """
    interaction_pairs: list[tuple[int, int]] = []
    thetas: list[float] = []
    phis: list[float] = []
    for c, s, i, j in givens_rotations:
        interaction_pairs.append((i, j))
        r, phi = cmath.polar(s)
        thetas.append(math.atan2(r, c))
        phis.append(phi)
    return interaction_pairs, thetas, phis


def _angles_to_rotations(
    interaction_pairs: list[tuple[int, int]],
    thetas,
    phis,
) -> list[tuple[float, complex, int, int]]:
    """Convert ``(pairs, thetas, phis)`` to ``(c, s, i, j)`` rotations.

    Uses ``c = cos(theta)``, ``s = sin(theta) * exp(1j * phi)``.
    """
    return [
        (math.cos(theta), cmath.rect(math.sin(theta), phi), i, j)
        for (i, j), theta, phi in zip(interaction_pairs, thetas, phis)
    ]


def _compute_n_keep(
    layer_ids: list[int],
    n_existing: int,
    max_givens: int | None,
    max_layers: int | None,
) -> int:
    """Determine how many rotations to keep given the compression caps.

    We start from the existing exact decomposition and only trim from the end, so the
    ceiling is ``n_existing``. Assumes ``layer_ids`` is in layer order.
    """
    n_keep = n_existing
    if max_layers is not None:
        n_keep = sum(1 for layer_id in layer_ids if layer_id < max_layers)
    if max_givens is not None:
        n_keep = min(n_keep, max_givens)
    return n_keep


def _run_optimizer(
    value_and_grad,
    target: jax.Array,
    x0: np.ndarray,
    optimize_kwargs: dict,
) -> np.ndarray:
    """Minimize the compression objective and return the optimal variable vector.

    Wraps the jitted ``value_and_grad`` (which takes ``(x, target)``) in the
    float/ndarray interface :func:`scipy.optimize.minimize` expects, defaulting to
    L-BFGS-B.
    """

    def scipy_func(x: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = value_and_grad(jnp.asarray(x), target)
        return float(value), np.asarray(grad)

    optimize_kwargs.setdefault("method", "L-BFGS-B")
    result = scipy.optimize.minimize(scipy_func, x0, jac=True, **optimize_kwargs)
    return result.x


def _reconstruct_orbital_rotation_jax(
    thetas: jax.Array,
    phis: jax.Array,
    phase_angles: jax.Array,
    interaction_pairs: list[tuple[int, int]],
) -> jax.Array:
    """Reconstruct an orbital rotation from Givens angles (JAX, differentiable).

    Reconstructs :math:`U = D G_L^* \\cdots G_1^*` where the Givens rotations are
    given by their angles and applied in reverse order as column operations.
    """
    mat = jnp.diag(jnp.exp(1j * phase_angles))
    for (i, j), theta, phi in zip(interaction_pairs[::-1], thetas[::-1], phis[::-1]):
        c = jnp.cos(theta)
        s = jnp.sin(theta) * jnp.exp(1j * phi)
        col_j = mat[:, j]
        col_i = mat[:, i]
        # Right-multiply by conj(G_k), matching zrot(col_j, col_i, c, conj(s)).
        new_col_j = c * col_j + jnp.conj(s) * col_i
        new_col_i = c * col_i - s * col_j
        mat = mat.at[:, j].set(new_col_j)
        mat = mat.at[:, i].set(new_col_i)
    return mat


@functools.cache
def _make_compressed_value_and_grad(
    n: int, pairs_kept: tuple[tuple[int, int], ...], n_keep: int
):
    """Build a jitted value-and-gradient function for the compression objective.

    The result is cached and reused across optimizer iterations and across calls
    with the same static structure ``(n, pairs_kept, n_keep)``, so the loss is traced
    and compiled only once per structure. The target unitary is passed as a runtime
    argument (it changes every call), and the gradient is taken with respect to the
    flat variable vector ``x`` only.

    The loss is the Hilbert-Schmidt infidelity
    :math:`1 - \\lvert \\mathrm{Tr}(U^\\dagger V) \\rvert^2 / n^2`, where :math:`U`
    (``target``) is the exact :math:`n \\times n` unitary and :math:`V` is the
    reconstructed one. Unlike the Frobenius distance, this objective is invariant to a
    global phase between :math:`U` and :math:`V`.
    """

    def loss(x: jax.Array, target: jax.Array) -> jax.Array:
        thetas_x = x[:n_keep]
        phis_x = x[n_keep : 2 * n_keep]
        phase_angles_x = x[2 * n_keep :]
        reconstructed = _reconstruct_orbital_rotation_jax(
            thetas_x, phis_x, phase_angles_x, list(pairs_kept)
        )
        overlap = jnp.sum(jnp.conj(target) * reconstructed)
        return 1.0 - jnp.abs(overlap / n) ** 2

    return jax.jit(jax.value_and_grad(loss, argnums=0))


def _givens_decomposition_compressed(
    mat: np.ndarray,
    tol: float = 1e-12,
    max_givens: int | None = None,
    max_layers: int | None = None,
    **optimize_kwargs,
) -> tuple[list[tuple[float, complex, int, int]], np.ndarray]:
    r"""Compressed Givens rotation decomposition of a unitary matrix.

    This function computes an approximate Givens rotation decomposition of a unitary
    matrix using at most a specified number of Givens rotations or brickwork layers.
    See :func:`givens_decomposition` for a description of the decomposition and the
    brickwork pattern.

    This function starts from the exact decomposition (respecting ``tol``) and never
    adds Givens rotations beyond those already present. The ``max_givens`` and
    ``max_layers`` constraints only *trim* rotations from the end of the brickwork
    pattern: at most ``max_givens`` rotations are retained, or only those lying in the
    first ``max_layers`` brickwork layers. If neither constraint trims anything, the
    exact decomposition is returned unchanged (so a near-identity matrix may use fewer
    rotations than the budget). Otherwise the angles of the retained rotations
    (together with the diagonal phases) are numerically optimized to minimize the
    Hilbert-Schmidt infidelity to the original unitary matrix, which ignores the global
    phase of the reconstructed matrix; in this case ``tol`` is not respected, and the
    reconstructed matrix may differ from ``mat`` by more than ``tol``.

    Args:
        mat: The unitary matrix to decompose into Givens rotations.
        tol: Matrix entries smaller than this value will be treated as equal to zero
            when computing the exact decomposition that this function trims.
        max_givens: The maximum number of Givens rotations to use. If ``None``, no
            limit is imposed on the number of Givens rotations.
        max_layers: The maximum number of brickwork layers to use. If ``None``, no
            limit is imposed on the number of layers.
        optimize_kwargs: Keyword arguments to pass to :func:`scipy.optimize.minimize`.

    Returns:
        - A list containing the Givens rotations, each represented as a 4-tuple
          :math:`(c, s, i, j)`.
        - A Numpy array containing the diagonal elements of the matrix :math:`D`.
    """
    _validate_caps(max_givens, max_layers)

    mat = mat.astype(complex, copy=False)

    # Compute the exact decomposition, respecting tol. This is the starting point:
    # we never add rotations beyond the ones already present here, we only trim.
    givens_rotations, phases = _lib.givens_decomposition(mat, tol=tol)
    n_existing = len(givens_rotations)
    interaction_pairs, thetas, phis = _rotations_to_angles(givens_rotations)
    layer_ids = _greedy_layer_ids(interaction_pairs)

    # Reorder the rotations into layer order so that a prefix of the list corresponds
    # to whole brickwork layers. The stable (layer, original position) sort preserves
    # a valid application order (rotations within a layer act on disjoint orbitals, and
    # each rotation still follows the ones it depends on). Without this, the returned
    # decomposition could schedule to more layers than intended.
    order = sorted(range(n_existing), key=lambda k: (layer_ids[k], k))
    interaction_pairs = [interaction_pairs[k] for k in order]
    thetas = [thetas[k] for k in order]
    phis = [phis[k] for k in order]
    layer_ids = [layer_ids[k] for k in order]

    n_keep = _compute_n_keep(layer_ids, n_existing, max_givens, max_layers)

    # If nothing is dropped, return the exact decomposition. It is returned in
    # brickwork/layer order (not the raw _lib order) so that its scheduled circuit
    # depth respects the brickwork layout, matching what the layer budget promises.
    if n_keep >= n_existing:
        rotations = _angles_to_rotations(interaction_pairs, thetas, phis)
        return rotations, phases

    pairs_kept = tuple(interaction_pairs[:n_keep])
    thetas0 = np.array(thetas[:n_keep])
    phis0 = np.array(phis[:n_keep])
    phase_angles0 = np.angle(phases)

    target = jnp.asarray(mat)

    # Reuse a jitted value-and-gradient function cached by static structure, so the
    # loss is compiled once and reused across all optimizer iterations (and across
    # calls with the same structure) instead of being re-traced eagerly each step.
    value_and_grad = _make_compressed_value_and_grad(mat.shape[0], pairs_kept, n_keep)

    x0 = np.concatenate([thetas0, phis0, phase_angles0])
    x_opt = _run_optimizer(value_and_grad, target, x0, optimize_kwargs)

    thetas_opt = x_opt[:n_keep]
    phis_opt = x_opt[n_keep : 2 * n_keep]
    phase_angles_opt = x_opt[2 * n_keep :]

    rotations = _angles_to_rotations(list(pairs_kept), thetas_opt, phis_opt)
    diagonal = np.exp(1j * phase_angles_opt)
    return rotations, diagonal


def _reconstruct_slater_isometry_jax(
    thetas: jax.Array,
    phis: jax.Array,
    interaction_pairs: list[tuple[int, int]],
    m: int,
    n: int,
) -> jax.Array:
    """Reconstruct the occupied orbitals of a Slater determinant (JAX, differentiable).

    Starts from the isometry of the electronic configuration with the first ``m``
    orbitals occupied (``eye(m, n)``) and applies the Givens rotations as column
    operations, matching the state-preparation circuit produced by
    :func:`givens_decomposition_slater`. The reconstructed rows remain orthonormal by
    construction.
    """
    mat = jnp.eye(m, n, dtype=complex)
    for (i, j), theta, phi in zip(interaction_pairs, thetas, phis):
        c = jnp.cos(theta)
        s = jnp.sin(theta) * jnp.exp(1j * phi)
        col_i = mat[:, i]
        col_j = mat[:, j]
        new_col_j = c * col_j - s * col_i
        new_col_i = c * col_i + jnp.conj(s) * col_j
        mat = mat.at[:, j].set(new_col_j)
        mat = mat.at[:, i].set(new_col_i)
    return mat


@functools.cache
def _make_compressed_slater_value_and_grad(
    m: int, n: int, pairs_kept: tuple[tuple[int, int], ...], n_keep: int
):
    """Build a jitted value-and-gradient function for the Slater compression objective.

    Mirrors :func:`_make_compressed_value_and_grad` but optimizes the fidelity of the
    prepared Slater determinant rather than the Frobenius distance of a full unitary.
    The loss is :math:`1 - \\lvert \\det(A B^\\dagger) \\rvert^2`, where :math:`A` is
    the reconstructed occupied-orbital isometry and :math:`B` (``target``) is the exact
    one. This is invariant to global phase and to rotations within the occupied space.
    """

    def loss(x: jax.Array, target: jax.Array) -> jax.Array:
        thetas_x = x[:n_keep]
        phis_x = x[n_keep:]
        reconstructed = _reconstruct_slater_isometry_jax(
            thetas_x, phis_x, list(pairs_kept), m, n
        )
        overlap = reconstructed @ jnp.conj(target).T
        return 1.0 - jnp.abs(jnp.linalg.det(overlap)) ** 2

    return jax.jit(jax.value_and_grad(loss, argnums=0))


def _givens_decomposition_slater_compressed(
    orbital_coeffs: np.ndarray,
    tol: float = 1e-12,
    max_givens: int | None = None,
    max_layers: int | None = None,
    **optimize_kwargs,
) -> list[tuple[float, complex, int, int]]:
    r"""Compressed Givens rotation decomposition for Slater determinant preparation.

    This function computes an approximate state-preparation decomposition using at most
    a specified number of Givens rotations or layers. See
    :func:`givens_decomposition_slater` for a description of the decomposition. It
    starts from the exact decomposition (respecting ``tol``) and never adds Givens
    rotations beyond those already present. The ``max_givens`` and ``max_layers``
    constraints only *trim* rotations from the end of the pattern: at most
    ``max_givens`` rotations are retained, or only those lying in the first
    ``max_layers`` layers. If neither constraint trims anything, the exact
    decomposition is returned unchanged (so a near-identity matrix may use fewer
    rotations than the budget). Otherwise the angles of the retained rotations are
    numerically optimized to maximize the fidelity of the prepared Slater determinant
    with the target; in this case ``tol`` is not respected, and the prepared state may
    differ from the target by more than ``tol``.

    Args:
        orbital_coeffs: The :math:`m \times n` matrix of occupied orbital coefficients.
        tol: Matrix entries smaller than this value will be treated as equal to zero
            when computing the exact decomposition that this function trims.
        max_givens: The maximum number of Givens rotations to use. If ``None``, no limit
            is imposed on the number of Givens rotations.
        max_layers: The maximum number of layers to use. If ``None``, no limit is
            imposed on the number of layers.
        optimize_kwargs: Keyword arguments to pass to :func:`scipy.optimize.minimize`.

    Returns:
        A list containing the Givens rotations, each represented as a 4-tuple
        :math:`(c, s, i, j)`.
    """
    _validate_caps(max_givens, max_layers)

    orbital_coeffs = orbital_coeffs.astype(complex, copy=False)
    m, n = orbital_coeffs.shape

    # Compute the exact decomposition (respecting tol) and its diamond layout. This
    # is the starting point: we never add rotations beyond the ones already present
    # here, we only trim.
    givens_rotations = _lib.givens_decomposition_slater(orbital_coeffs, tol)
    interaction_pairs, thetas, phis = _rotations_to_angles(givens_rotations)
    layer_ids = _greedy_layer_ids(interaction_pairs)
    n_existing = len(interaction_pairs)

    # Reorder the rotations into layer order so that a prefix of the list corresponds
    # to whole parallel layers. The exact decomposition returns rotations in a valid
    # but interleaved diamond order, so ``layer_ids`` is not monotonic; sorting by
    # (layer, original position) is stable and preserves a valid application order
    # (all rotations in a layer act on disjoint orbitals, and each rotation still
    # follows the ones it depends on). Without this, trimming a list-order prefix
    # would keep the wrong rotations and inflate the circuit depth.
    order = sorted(range(n_existing), key=lambda k: (layer_ids[k], k))
    interaction_pairs = [interaction_pairs[k] for k in order]
    thetas = [thetas[k] for k in order]
    phis = [phis[k] for k in order]
    layer_ids = [layer_ids[k] for k in order]

    n_keep = _compute_n_keep(layer_ids, n_existing, max_givens, max_layers)

    # If nothing is dropped, return the exact (tol-respecting) decomposition as-is.
    if n_keep >= n_existing:
        return givens_rotations

    pairs_kept = tuple(interaction_pairs[:n_keep])
    thetas0 = np.array(thetas[:n_keep])
    phis0 = np.array(phis[:n_keep])

    target = jnp.asarray(orbital_coeffs)

    value_and_grad = _make_compressed_slater_value_and_grad(m, n, pairs_kept, n_keep)

    x0 = np.concatenate([thetas0, phis0])
    x_opt = _run_optimizer(value_and_grad, target, x0, optimize_kwargs)

    thetas_opt = x_opt[:n_keep]
    phis_opt = x_opt[n_keep:]

    return _angles_to_rotations(list(pairs_kept), thetas_opt, phis_opt)
