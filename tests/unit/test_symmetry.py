"""Unit tests for symmetry.py — no optional backends required."""

# Copyright (c) Alexander Bonkowski
# Distributed under the terms of the MIT License
# author: Alexander Bonkowski

from __future__ import annotations

from unittest import TestCase

import numpy as np
import pytest

from agility.symmetry import tilt_twist_decomposition


def _rotation_quat(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Return a scalar-last unit quaternion for a rotation by *angle_deg* around *axis*."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = np.radians(angle_deg) / 2.0
    return np.array([*(axis * np.sin(half)), np.cos(half)])


@pytest.mark.unit
class TestTiltTwistDecomposition(TestCase):
    """Test tilt_twist_decomposition with analytically verifiable cases."""

    # Identity quaternion [x, y, z, w] = [0, 0, 0, 1]
    _Q_IDENTITY = np.array([[0.0, 0.0, 0.0, 1.0]])
    # Boundary plane normal along z
    _NORMAL_Z = np.array([0.0, 0.0, 1.0])

    def _pair(
        self,
        axis: list[float],
        angle_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (q_i, q_j) where q_i is the identity and q_j is the given rotation."""
        q_j = _rotation_quat(axis, angle_deg)
        return self._Q_IDENTITY, q_j[None, :]

    # ------------------------------------------------------------------
    # Pure tilt: rotation axis in the boundary plane (perpendicular to m)
    # ------------------------------------------------------------------

    def test_pure_tilt_x_axis(self) -> None:
        """30° rotation around x with boundary normal z → tilt=30°, twist=0°."""
        q_i, q_j = self._pair([1, 0, 0], 30.0)
        tilt, twist = tilt_twist_decomposition(q_i, q_j, self._NORMAL_Z)
        np.testing.assert_allclose(tilt, [30.0], atol=1e-10)
        np.testing.assert_allclose(twist, [0.0], atol=1e-10)

    def test_pure_tilt_y_axis(self) -> None:
        """45° rotation around y with boundary normal z → tilt=45°, twist=0°."""
        q_i, q_j = self._pair([0, 1, 0], 45.0)
        tilt, twist = tilt_twist_decomposition(q_i, q_j, self._NORMAL_Z)
        np.testing.assert_allclose(tilt, [45.0], atol=1e-10)
        np.testing.assert_allclose(twist, [0.0], atol=1e-10)

    # ------------------------------------------------------------------
    # Pure twist: rotation axis parallel to boundary normal
    # ------------------------------------------------------------------

    def test_pure_twist_z_axis(self) -> None:
        """30° rotation around z with boundary normal z → tilt=0°, twist=30°."""
        q_i, q_j = self._pair([0, 0, 1], 30.0)
        tilt, twist = tilt_twist_decomposition(q_i, q_j, self._NORMAL_Z)
        np.testing.assert_allclose(tilt, [0.0], atol=1e-10)
        np.testing.assert_allclose(twist, [30.0], atol=1e-10)

    def test_pure_twist_negative_normal(self) -> None:
        """Negating the boundary normal must not change the result (normal is normalised)."""
        q_i, q_j = self._pair([0, 0, 1], 60.0)
        tilt_pos, twist_pos = tilt_twist_decomposition(q_i, q_j, [0.0, 0.0, 1.0])
        tilt_neg, twist_neg = tilt_twist_decomposition(q_i, q_j, [0.0, 0.0, -1.0])
        np.testing.assert_allclose(tilt_neg, tilt_pos, atol=1e-10)
        np.testing.assert_allclose(twist_neg, twist_pos, atol=1e-10)

    # ------------------------------------------------------------------
    # Identity (zero misorientation)
    # ------------------------------------------------------------------

    def test_identity_misorientation(self) -> None:
        """q_i == q_j must yield tilt=0° and twist=0°."""
        q = np.array([[0.0, 0.0, 0.0, 1.0]])
        tilt, twist = tilt_twist_decomposition(q, q, self._NORMAL_Z)
        np.testing.assert_allclose(tilt, [0.0], atol=1e-10)
        np.testing.assert_allclose(twist, [0.0], atol=1e-10)

    # ------------------------------------------------------------------
    # Symmetry of decomposition
    # ------------------------------------------------------------------

    def test_swap_grains_same_angles(self) -> None:
        """Swapping q_i and q_j must yield the same tilt and twist angles."""
        q_i, q_j = self._pair([1, 0, 0], 40.0)
        tilt_fwd, twist_fwd = tilt_twist_decomposition(q_i, q_j, self._NORMAL_Z)
        tilt_rev, twist_rev = tilt_twist_decomposition(q_j, q_i, self._NORMAL_Z)
        np.testing.assert_allclose(tilt_rev, tilt_fwd, atol=1e-10)
        np.testing.assert_allclose(twist_rev, twist_fwd, atol=1e-10)

    # ------------------------------------------------------------------
    # Mixed boundary: axis at 45° to boundary normal
    # ------------------------------------------------------------------

    def test_mixed_boundary_equal_tilt_twist(self) -> None:
        """Axis at 45° to boundary normal → tilt == twist."""
        axis = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
        q_i, q_j = self._pair(axis, 30.0)
        tilt, twist = tilt_twist_decomposition(q_i, q_j, self._NORMAL_Z)
        np.testing.assert_allclose(tilt, twist, atol=1e-10)

    def test_mixed_boundary_angles_less_than_total(self) -> None:
        """Both tilt and twist angles must be strictly less than the total rotation angle."""
        axis = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
        angle = 60.0
        q_i, q_j = self._pair(axis, angle)
        tilt, twist = tilt_twist_decomposition(q_i, q_j, self._NORMAL_Z)
        assert float(tilt[0]) < angle
        assert float(twist[0]) < angle

    # ------------------------------------------------------------------
    # Batch (N > 1) inputs
    # ------------------------------------------------------------------

    def test_batch_consistent_with_single(self) -> None:
        """Batched evaluation must match individual single-pair results."""
        angles = [15.0, 30.0, 45.0]
        axes = [[1, 0, 0], [0, 0, 1], [1, 0, 1]]
        q_i_list = []
        q_j_list = []
        for ax, ang in zip(axes, angles, strict=True):
            q_j_list.append(_rotation_quat(ax, ang))
            q_i_list.append([0.0, 0.0, 0.0, 1.0])
        q_i_batch = np.array(q_i_list)
        q_j_batch = np.array(q_j_list)
        tilt_batch, twist_batch = tilt_twist_decomposition(q_i_batch, q_j_batch, self._NORMAL_Z)

        for k, (ax, ang) in enumerate(zip(axes, angles, strict=True)):
            q_i_s = np.array([[0.0, 0.0, 0.0, 1.0]])
            q_j_s = np.array([_rotation_quat(ax, ang)])
            tilt_s, twist_s = tilt_twist_decomposition(q_i_s, q_j_s, self._NORMAL_Z)
            np.testing.assert_allclose(tilt_batch[k], tilt_s[0], atol=1e-10)
            np.testing.assert_allclose(twist_batch[k], twist_s[0], atol=1e-10)

    # ------------------------------------------------------------------
    # Per-pair normals
    # ------------------------------------------------------------------

    def test_per_pair_normals(self) -> None:
        """Per-pair normals of shape (N, 3) must give the same result as per-call (3,)."""
        # Pure tilt pair
        q_i = np.array([[0.0, 0.0, 0.0, 1.0]])
        q_j = np.array([_rotation_quat([1, 0, 0], 30.0)])
        tilt_single, twist_single = tilt_twist_decomposition(q_i, q_j, [0.0, 0.0, 1.0])
        # Use per-pair normal array of shape (1, 3)
        tilt_batch, twist_batch = tilt_twist_decomposition(
            q_i,
            q_j,
            np.array([[0.0, 0.0, 1.0]]),
        )
        np.testing.assert_allclose(tilt_batch, tilt_single, atol=1e-10)
        np.testing.assert_allclose(twist_batch, twist_single, atol=1e-10)

    # ------------------------------------------------------------------
    # Unnormalised boundary normal must be accepted
    # ------------------------------------------------------------------

    def test_unnormalised_boundary_normal_accepted(self) -> None:
        """Scaled boundary normals must produce the same result as unit normals."""
        q_i, q_j = self._pair([1, 0, 0], 30.0)
        tilt_unit, twist_unit = tilt_twist_decomposition(q_i, q_j, [0.0, 0.0, 1.0])
        tilt_scaled, twist_scaled = tilt_twist_decomposition(q_i, q_j, [0.0, 0.0, 5.0])
        np.testing.assert_allclose(tilt_scaled, tilt_unit, atol=1e-10)
        np.testing.assert_allclose(twist_scaled, twist_unit, atol=1e-10)

    def test_zero_boundary_normal_raises(self) -> None:
        """A zero boundary-normal vector must raise ValueError."""
        q_i, q_j = self._pair([1, 0, 0], 30.0)
        with pytest.raises(ValueError, match="non-zero vector"):
            tilt_twist_decomposition(q_i, q_j, [0.0, 0.0, 0.0])

    def test_zero_row_in_per_pair_boundary_normals_raises(self) -> None:
        """Any zero row in per-pair boundary normals must raise ValueError."""
        q_i = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
        q_j = np.array(
            [
                _rotation_quat([1, 0, 0], 30.0),
                _rotation_quat([0, 0, 1], 30.0),
            ],
        )
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
        )
        with pytest.raises(ValueError, match="rows must be non-zero vectors"):
            tilt_twist_decomposition(q_i, q_j, normals)

    def test_optional_cubic_symmetry_reduction(self) -> None:
        """Optional internal cubic symmetry reduction can collapse symmetry-equivalent pairs."""
        q_i = np.array([[0.0, 0.0, 0.0, 1.0]])
        q_j = np.array([_rotation_quat([0, 0, 1], 90.0)])
        tilt_raw, twist_raw = tilt_twist_decomposition(q_i, q_j, self._NORMAL_Z)
        tilt_red, twist_red = tilt_twist_decomposition(
            q_i,
            q_j,
            self._NORMAL_Z,
            reduce_cubic_symmetry=True,
        )
        np.testing.assert_allclose(tilt_raw, [0.0], atol=1e-10)
        np.testing.assert_allclose(twist_raw, [90.0], atol=1e-10)
        np.testing.assert_allclose(tilt_red, [0.0], atol=1e-10)
        np.testing.assert_allclose(twist_red, [0.0], atol=1e-10)
