"""Integration tests for plotting functions — requires ovito to be installed."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from unittest import TestCase

import pytest

from agility.analysis import GBStructure
from agility.plotting import plot_mdf, render_ovito

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILES_DIR = MODULE_DIR.parent / "files"


@pytest.mark.integration
@pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
class TestPlotting(TestCase):
    """Test ovito rendering via the plotting module."""

    def setUp(self) -> None:
        """Set up the test."""
        self.data = GBStructure("ovito", f"{TEST_FILES_DIR}/aluminium.lmp")

        assert self.data is not None

    def test_render_ovito(self) -> None:
        """Test the render_ovito function returns a QImage with expected dimensions."""
        image = render_ovito(self.data.pipeline)
        from PySide6.QtGui import QImage  # noqa: PLC0415

        assert isinstance(image, QImage)
        assert image.width() == 282
        assert image.height() == 262

    def test_plot_mdf_cubic_symmetry(self) -> None:
        """Test MDF plotting with cubic symmetry reduction on ovito grain orientations."""
        import matplotlib.figure  # noqa: PLC0415

        self.data.perform_ptm(enabled=("fcc",), output_orientation=True)
        orientations = self.data.get_distinct_grains()
        assert orientations is not None

        fig = plot_mdf(orientations, symmetry="cubic")
        assert isinstance(fig, matplotlib.figure.Figure)
        assert "cubic disorientation" in fig.axes[0].get_title().lower()
