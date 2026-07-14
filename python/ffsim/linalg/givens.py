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
import itertools
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
    ``max_givens`` or ``max_layers`` is specified, the decomposition is *compressed*
    to use at most that many Givens rotations or brickwork layers, respectively. In
    this case, the Givens rotations at the beginning of the brickwork pattern are
    retained, and their angles (together with the diagonal phases) are numerically
    optimized to minimize the Frobenius distance :math:`\lVert U - V \rVert_F`
    between the original matrix :math:`U` and the reconstructed matrix :math:`V`.
    The returned decomposition is then only approximate. When both ``max_givens`` and
    ``max_layers`` are given, the tighter of the two constraints is applied.

    References:
        - `Clements et al., "Optimal design for universal multiport interferometers" (2016)`_

    Args:
        mat: The unitary matrix to decompose into Givens rotations.
        tol: Matrix entries smaller than this value will be treated as equal to zero.
            Only used for the exact (uncompressed) decomposition.
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


def _brickwork_givens_rotations(
    interaction_pairs: list[tuple[int, int]],
    thetas: list[float],
    phis: list[float],
    norb: int,
) -> tuple[list[tuple[int, int]], list[float], list[float], list[int]]:
    """Expand a sparse Givens rotation decomposition to a full brickwork pattern.

    Returns the interaction pairs, thetas, and phis reordered into a brickwork
    pattern, along with a parallel list giving the brickwork layer index of each
    Givens rotation. Layers are indexed in the order they are applied, starting
    from zero. The full brickwork pattern has ``norb`` layers.
    """
    # Construct a brickwork pattern of Givens rotations with angles set to zero
    q, r = divmod(norb, 2)
    even_layers = [
        [((i, i + 1), 0.0, 0.0) for i in range(0, norb - 1, 2)] for _ in range(q + r)
    ]
    odd_layers = [
        [((i, i + 1), 0.0, 0.0) for i in range(1, norb - 1, 2)] for _ in range(q)
    ]
    # even_layer_index[i] is the index of the last even layer acting on orbital i
    even_layer_index = [-1] * norb
    # odd_layer_index[i] is the index of the last odd layer acting on orbital i
    odd_layer_index = [-1] * norb
    for (i, j), theta, phi in zip(interaction_pairs, thetas, phis):
        if i > j:
            # Enforce i < j
            i, j = j, i
            theta = -theta
            phi = -phi
        if i % 2 == 0:
            # Even layer
            # Get the index of the even layer this Givens rotation should go in
            index = (
                max(
                    even_layer_index[i],
                    even_layer_index[j],
                    odd_layer_index[i],
                    odd_layer_index[j],
                )
                + 1
            )
            # Add the Givens rotation in the appropriate place
            even_layers[index][i // 2] = ((i, j), theta, phi)
            # Update the even layer index
            even_layer_index[i] = index
            even_layer_index[j] = index
        else:
            # Odd layer
            # Get the index of the odd layer this Givens rotation should go in
            index = max(
                odd_layer_index[i] + 1,
                odd_layer_index[j] + 1,
                even_layer_index[i],
                even_layer_index[j],
            )
            # Add the Givens rotation in the appropriate place
            odd_layers[index][i // 2] = ((i, j), theta, phi)
            # Update the odd layer index
            odd_layer_index[i] = index
            odd_layer_index[j] = index
    # Construct the new Givens rotation decomposition and return
    new_interaction_pairs = []
    new_thetas = []
    new_phis = []
    layer_ids = []
    # The applied layers alternate even, odd, even, odd, ...; assign each nonempty
    # layer the next brickwork layer index.
    layer_id = 0
    for even_layer, odd_layer in itertools.zip_longest(
        even_layers, odd_layers, fillvalue=()
    ):
        for layer in [even_layer, odd_layer]:
            if not layer:
                continue
            for pair, theta, phi in layer:
                new_interaction_pairs.append(pair)
                new_thetas.append(theta)
                new_phis.append(phi)
                layer_ids.append(layer_id)
            layer_id += 1
    return new_interaction_pairs, new_thetas, new_phis, layer_ids


def _reconstruct_orbital_rotation_jax(
    thetas: jax.Array,
    phis: jax.Array,
    phase_angles: jax.Array,
    interaction_pairs: list[tuple[int, int]],
    norb: int,
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

    The exact decomposition uses up to :math:`\frac{n(n-1)}{2}` Givens rotations
    arranged in :math:`n` brickwork layers. This function keeps only the Givens
    rotations at the beginning of the brickwork pattern, subject to the ``max_givens``
    and ``max_layers`` constraints, and optimizes the angles of the retained rotations
    (together with the diagonal phases) to minimize the Frobenius distance to the
    original unitary matrix.

    Args:
        mat: The unitary matrix to decompose into Givens rotations.
        tol: Matrix entries smaller than this value will be treated as equal to zero
            when nothing is dropped and the exact decomposition is returned.
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
    if max_givens is not None and max_givens < 0:
        raise ValueError(f"max_givens must be non-negative. Got {max_givens}.")
    if max_layers is not None and max_layers < 0:
        raise ValueError(f"max_layers must be non-negative. Got {max_layers}.")

    mat = mat.astype(complex, copy=False)
    norb, _ = mat.shape
    max_full = norb * (norb - 1) // 2

    # Compute the full brickwork layout from the exact decomposition.
    givens_rotations, phases = _lib.givens_decomposition(mat, tol=0.0)
    interaction_pairs: list[tuple[int, int]] = []
    thetas: list[float] = []
    phis: list[float] = []
    for c, s, i, j in givens_rotations:
        interaction_pairs.append((i, j))
        r, phi = cmath.polar(s)
        thetas.append(math.atan2(r, c))
        phis.append(phi)
    interaction_pairs, thetas, phis, layer_ids = _brickwork_givens_rotations(
        interaction_pairs, thetas, phis, norb=norb
    )

    # Determine the number of Givens rotations to keep.
    n_keep = max_full
    if max_layers is not None:
        n_keep = sum(1 for layer_id in layer_ids if layer_id < max_layers)
    if max_givens is not None:
        n_keep = min(n_keep, max_givens)

    # If nothing is dropped, return the exact decomposition.
    if n_keep >= max_full:
        return _lib.givens_decomposition(mat, tol=tol)

    pairs_kept = interaction_pairs[:n_keep]
    thetas0 = np.array(thetas[:n_keep])
    phis0 = np.array(phis[:n_keep])
    phase_angles0 = np.angle(phases)

    target = jnp.asarray(mat)

    def loss(x: jax.Array) -> jax.Array:
        thetas_x = x[:n_keep]
        phis_x = x[n_keep : 2 * n_keep]
        phase_angles_x = x[2 * n_keep :]
        reconstructed = _reconstruct_orbital_rotation_jax(
            thetas_x, phis_x, phase_angles_x, pairs_kept, norb
        )
        return jnp.sum(jnp.abs(target - reconstructed) ** 2)

    value_and_grad = jax.value_and_grad(loss)

    def scipy_func(x: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = value_and_grad(jnp.asarray(x))
        return float(value), np.asarray(grad)

    x0 = np.concatenate([thetas0, phis0, phase_angles0])
    optimize_kwargs.setdefault("method", "L-BFGS-B")
    result = scipy.optimize.minimize(scipy_func, x0, jac=True, **optimize_kwargs)

    thetas_opt = result.x[:n_keep]
    phis_opt = result.x[n_keep : 2 * n_keep]
    phase_angles_opt = result.x[2 * n_keep :]

    rotations = [
        (
            math.cos(theta),
            cmath.rect(math.sin(theta), phi),
            i,
            j,
        )
        for (i, j), theta, phi in zip(pairs_kept, thetas_opt, phis_opt)
    ]
    diagonal = np.exp(1j * phase_angles_opt)
    return rotations, diagonal
