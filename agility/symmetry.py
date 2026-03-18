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
