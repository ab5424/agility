"""Unit tests for plotting.py using mock data — no real ovito pipeline required."""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import pytest

from agility.plotting import plot_face_order, plot_mdf


@pytest.mark.unit
class TestPlotFaceOrder(TestCase):
    """Test the plot_face_order function with mock ovito DataCollection objects."""

    def _make_mock_data(
        self,
        identifiers: list[int],
        face_orders: list[int],
        prop: str = "Max Face Order",
    ) -> MagicMock:
        """Return a mock DataCollection whose particles behave like a dict."""
        mock_data = MagicMock()
        mock_data.particles.__getitem__.side_effect = lambda key: {
            "Particle Identifier": identifiers,
            prop: face_orders,
        }[key]
        return mock_data

    def test_returns_figure(self) -> None:
        """Test that plot_face_order returns a matplotlib Figure."""
        import matplotlib.figure  # noqa: PLC0415

        mock_data = self._make_mock_data([1, 2, 3, 4], [4, 5, 6, 5])
        result = plot_face_order(mock_data)
        assert isinstance(result, matplotlib.figure.Figure)

    def test_custom_plot_property(self) -> None:
        """Test that a custom plot_property is used as the DataFrame column and axis."""
        import matplotlib.figure  # noqa: PLC0415

        prop = "Min Face Order"
        mock_data = self._make_mock_data([1, 2, 3], [2, 3, 2], prop=prop)
        result = plot_face_order(mock_data, plot_property=prop)
        assert isinstance(result, matplotlib.figure.Figure)

    def test_single_particle(self) -> None:
        """Test that plot_face_order handles a single-particle data collection."""
        import matplotlib.figure  # noqa: PLC0415

        mock_data = self._make_mock_data([1], [5])
        result = plot_face_order(mock_data)
        assert isinstance(result, matplotlib.figure.Figure)


@pytest.mark.unit
class TestPlotMdf(TestCase):
    """Test the plot_mdf function with synthetic quaternion data."""

    # Three distinct orientations used in most tests.
    # q1 = identity, q2 = 90° around x, q3 = 90° around z in scalar-last (x, y, z, w).
    _ORIENTATIONS: np.ndarray = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2],
            [0.0, 0.0, np.sqrt(2) / 2, np.sqrt(2) / 2],
        ],
    )

    def test_returns_figure(self) -> None:
        """Test that plot_mdf returns a matplotlib Figure."""
        import matplotlib.figure  # noqa: PLC0415

        result = plot_mdf(self._ORIENTATIONS)
        assert isinstance(result, matplotlib.figure.Figure)

    def test_custom_bins(self) -> None:
        """Test that the bins parameter is forwarded to the histogram."""
        import matplotlib.figure  # noqa: PLC0415

        result = plot_mdf(self._ORIENTATIONS, bins=10)
        assert isinstance(result, matplotlib.figure.Figure)
        ax = result.axes[0]
        assert len(ax.patches) == 10
        for patch in ax.patches:
            assert 0.0 <= patch.get_x() <= 180.0

    def test_density_false(self) -> None:
        """Test that density=False produces a count histogram."""
        import matplotlib.figure  # noqa: PLC0415

        result = plot_mdf(self._ORIENTATIONS, density=False)
        assert isinstance(result, matplotlib.figure.Figure)

    def test_known_misorientation_angle(self) -> None:
        """Verify that a 90° misorientation between two grains is histogrammed correctly."""
        # q_identity (0°) vs. q_90x (90° rotation around x-axis), scalar-last.
        q_identity = np.array([0.0, 0.0, 0.0, 1.0])
        q_90x = np.array([np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2])
        orientations = np.array([q_identity, q_90x])

        fig = plot_mdf(orientations, density=False)
        ax = fig.axes[0]

        non_empty = [p for p in ax.patches if p.get_height() > 0]
        assert len(non_empty) == 1
        bar = non_empty[0]
        # The single bar must span the 90° misorientation.
        assert bar.get_x() <= 90.0 <= bar.get_x() + bar.get_width()

    def test_unnormalised_quaternions_accepted(self) -> None:
        """Test that quaternions that are not already unit-length are normalised."""
        import matplotlib.figure  # noqa: PLC0415

        # Scale each row by a different factor; the function should still work.
        orientations = self._ORIENTATIONS * np.array([[2.0], [0.5], [3.0]])
        result = plot_mdf(orientations)
        assert isinstance(result, matplotlib.figure.Figure)

    def test_axes_labels(self) -> None:
        """Test that the returned figure has the expected axis labels."""
        result = plot_mdf(self._ORIENTATIONS)
        ax = result.axes[0]
        assert "Misorientation" in ax.get_title()
        assert "°" in ax.get_xlabel()

    def test_wrong_shape_raises(self) -> None:
        """Test that a non-(N, 4) array raises ValueError."""
        with pytest.raises(ValueError, match="orientations must have shape"):
            plot_mdf(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    def test_zero_norm_raises(self) -> None:
        """Test that a zero-norm quaternion raises ValueError."""
        orientations = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],  # zero norm
            ],
        )
        with pytest.raises(ValueError, match="zero-norm"):
            plot_mdf(orientations)

    def test_single_orientation_raises(self) -> None:
        """Test that fewer than 2 orientations raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            plot_mdf(np.array([[1.0, 0.0, 0.0, 0.0]]))

    def test_unsupported_symmetry_raises(self) -> None:
        """Test that unsupported symmetry names raise ValueError."""
        with pytest.raises(ValueError, match="unsupported symmetry"):
            plot_mdf(self._ORIENTATIONS, symmetry="hexagonal")

    def test_cubic_symmetry_reduces_misorientation(self) -> None:
        """Test cubic symmetry reduction changes 90° misorientation to 0° disorientation."""
        sqrt2_over_2 = 1.0 / np.sqrt(2.0)
        q_identity = np.array([0.0, 0.0, 0.0, 1.0])
        q_90z = np.array([0.0, 0.0, sqrt2_over_2, sqrt2_over_2])
        orientations = np.array([q_identity, q_90z])

        fig_raw = plot_mdf(orientations, bins=30, density=False)
        raw_ax = fig_raw.axes[0]
        raw_bar = next(p for p in raw_ax.patches if p.get_height() > 0)
        raw_center = raw_bar.get_x() + raw_bar.get_width() / 2

        fig_sym = plot_mdf(orientations, bins=30, density=False, symmetry="cubic")
        sym_ax = fig_sym.axes[0]
        sym_bar = next(p for p in sym_ax.patches if p.get_height() > 0)
        sym_center = sym_bar.get_x() + sym_bar.get_width() / 2

        assert raw_center > 80.0
        assert sym_center < 5.0
