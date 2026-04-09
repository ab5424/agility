"""Symmetry utilities for orientation and misorientation analysis."""

# Copyright (c) Alexander Bonkowski
# Distributed under the terms of the MIT License
# author: Alexander Bonkowski

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two scalar-last quaternion arrays row-by-row.

    Args:
        a: Quaternion array with shape ``(N, 4)`` in ``(x, y, z, w)`` order.
        b: Quaternion array with shape ``(N, 4)`` in ``(x, y, z, w)`` order.

    Returns:
        Quaternion array with shape ``(N, 4)`` where row ``i`` equals
        ``a[i] * b[i]``.

    """
    av, aw = a[:, :3], a[:, 3:4]
    bv, bw = b[:, :3], b[:, 3:4]
    xyz = aw * bv + bw * av + np.cross(av, bv)
    w = aw * bw - np.sum(av * bv, axis=1, keepdims=True)
    return np.concatenate((xyz, w), axis=1)


@lru_cache(maxsize=1)
def _cubic_symmetry_quaternions() -> np.ndarray:
    """Return the 24 proper rotations of cubic ``m-3m`` as scalar-last quaternions."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    cubic_rot_mats = []
    for perm in itertools.permutations((0, 1, 2)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            mat = np.zeros((3, 3), dtype=float)
            for row, col in enumerate(perm):
                mat[row, col] = signs[row]
            if np.isclose(np.linalg.det(mat), 1.0):
                cubic_rot_mats.append(mat)
    result = Rotation.from_matrix(np.array(cubic_rot_mats)).as_quat()
    if len(result) != 24:
        msg = f"expected 24 cubic symmetry operators, got {len(result)}"
        raise RuntimeError(msg)
    result.flags.writeable = False
    return result


def cubic_disorientation_angles(q_i: np.ndarray, q_j: np.ndarray) -> np.ndarray:
    """Return cubic disorientation angles for paired scalar-last unit quaternions.

    Args:
        q_i: Unit quaternion array with shape ``(N, 4)`` in ``(x, y, z, w)`` order.
        q_j: Unit quaternion array with shape ``(N, 4)`` in ``(x, y, z, w)`` order.

    Returns:
        Array of shape ``(N,)`` with cubic disorientation angles in degrees.

    Note:
        This reduction evaluates all ``24 x 24`` left/right cubic symmetry
        combinations for each quaternion pair (``576`` symmetry evaluations per
        pair), with overall cost ``O(576 * N)`` for ``N`` pairs.

    """
    cubic_sym = _cubic_symmetry_quaternions()

    v_i = q_i[:, :3]
    w_i = q_i[:, 3:4]
    q_i_conj = np.concatenate((-v_i, w_i), axis=1)
    q_rel = _quat_mul(q_i_conj, q_j)

    max_abs_w = np.zeros(len(q_rel))
    for left in cubic_sym:
        q_left = _quat_mul(np.broadcast_to(left, q_rel.shape), q_rel)
        for right in cubic_sym:
            q_equiv = _quat_mul(q_left, np.broadcast_to(right, q_left.shape))
            max_abs_w = np.maximum(max_abs_w, np.abs(q_equiv[:, 3]))

    return np.degrees(2.0 * np.arccos(np.clip(max_abs_w, 0.0, 1.0)))


def tilt_twist_decomposition(
    q_i: np.ndarray,
    q_j: np.ndarray,
    boundary_normal: np.ndarray,
    *,
    reduce_cubic_symmetry: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Decompose misorientations into tilt and twist angle components.

    Given two grain orientation quaternions and a grain boundary plane normal, this
    function decomposes the misorientation into its tilt component (rotation axis lying
    in the boundary plane) and twist component (rotation axis perpendicular to the
    boundary plane).

    Args:
        q_i: Unit quaternion array with shape ``(N, 4)`` in ``(x, y, z, w)`` order
            representing the orientation of the first grain in each pair.
        q_j: Unit quaternion array with shape ``(N, 4)`` in ``(x, y, z, w)`` order
            representing the orientation of the second grain in each pair.
        boundary_normal: Grain boundary plane normal vector(s). Either a single vector
            of shape ``(3,)`` applied to all pairs, or an array of shape ``(N, 3)``
            with one normal per pair. Vectors are normalised internally.
        reduce_cubic_symmetry: If ``True``, apply internal cubic symmetry reduction to
            the misorientation before decomposition by searching ``24 x 24`` left/right
            symmetry-equivalent representations and selecting the minimum-angle
            disorientation representative.

    Returns:
        Tuple ``(tilt_angles, twist_angles)`` where both arrays have shape ``(N,)``
        and contain angles in degrees. The tilt angle is the
        rotation component whose axis lies in the boundary plane; the twist angle is
        the component whose axis is parallel to the boundary plane normal.

    Note:
        Given the misorientation rotation axis ``n`` and total angle ``θ``, the
        decomposition satisfies

        .. math::

            \\tan(\\theta_{\\text{tilt}}/2)
                = |\\mathbf{n}_{\\text{tilt}}| \\tan(\\theta/2)

            \\tan(\\theta_{\\text{twist}}/2)
                = |\\mathbf{n} \\cdot \\mathbf{m}| \\tan(\\theta/2)

        where ``m`` is the boundary plane normal and
        ``n_tilt = n - (n · m) m`` is the projection of ``n`` onto the boundary
        plane.

    """
    q_i = np.atleast_2d(np.asarray(q_i, dtype=float))
    q_j = np.atleast_2d(np.asarray(q_j, dtype=float))

    # Misorientation: q_rel = q_i^{-1} * q_j (conjugate of q_i times q_j)
    v_i = q_i[:, :3]
    w_i = q_i[:, 3:4]
    q_i_conj = np.concatenate((-v_i, w_i), axis=1)
    q_rel = _quat_mul(q_i_conj, q_j)

    if reduce_cubic_symmetry:
        cubic_sym = _cubic_symmetry_quaternions()
        max_abs_w = np.full(len(q_rel), -np.inf)
        q_best = np.empty_like(q_rel)
        for left in cubic_sym:
            q_left = _quat_mul(np.broadcast_to(left, q_rel.shape), q_rel)
            for right in cubic_sym:
                q_equiv = _quat_mul(q_left, np.broadcast_to(right, q_left.shape))
                abs_w = np.abs(q_equiv[:, 3])
                better = abs_w > max_abs_w
                if np.any(better):
                    q_best[better] = q_equiv[better]
                    max_abs_w[better] = abs_w[better]
        q_rel = q_best

    # Canonical form: ensure w >= 0 so that θ ∈ [0°, 180°]
    neg_mask = q_rel[:, 3] < 0
    q_rel[neg_mask] = -q_rel[neg_mask]

    v_rel = q_rel[:, :3]  # sin(θ/2) * n
    w_rel = q_rel[:, 3]  # cos(θ/2) >= 0

    # Normalise boundary normal(s)
    m = np.asarray(boundary_normal, dtype=float)
    if m.ndim == 1:
        norm = np.linalg.norm(m)
        if np.isclose(norm, 0.0):
            msg = "boundary_normal must be a non-zero vector"
            raise ValueError(msg)
        m = m / norm
        v_dot_m = v_rel @ m  # (N,)
        v_twist = v_dot_m[:, None] * m  # (N, 3)
    else:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        if np.any(np.isclose(norms, 0.0)):
            msg = "boundary_normal rows must be non-zero vectors"
            raise ValueError(msg)
        m = m / norms
        v_dot_m = np.sum(v_rel * m, axis=1)  # (N,)
        v_twist = v_dot_m[:, None] * m  # (N, 3)

    v_tilt = v_rel - v_twist  # (N, 3)

    # Component magnitudes: |v_tilt| = |n_tilt| * sin(θ/2), etc.
    tilt_sin_half = np.linalg.norm(v_tilt, axis=1)  # (N,)
    twist_sin_half = np.abs(v_dot_m)  # (N,)

    # θ_component = 2 * arctan2(|v_component|, cos(θ/2))
    tilt_angles = np.degrees(2.0 * np.arctan2(tilt_sin_half, w_rel))
    twist_angles = np.degrees(2.0 * np.arctan2(twist_sin_half, w_rel))

    return tilt_angles, twist_angles
