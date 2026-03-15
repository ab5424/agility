"""Unit tests for plotting.py using mock data — no real ovito pipeline required."""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import MagicMock

import pytest

from agility.plotting import plot_face_order


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
