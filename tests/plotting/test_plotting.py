"""Test the plotting functions."""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase

try:
    from PySide6.QtGui import QImage
except ImportError:
    QImage = None

from agility.analysis import GBStructure
from agility.plotting import render_ovito

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILES_DIR = Path(MODULE_DIR / ".." / ".." / "tests" / "files")


class TestPlotting(TestCase):
    """Test the Plotting class."""

    def setUp(self) -> None:
        """Set up the test."""
        self.data = GBStructure("ovito", f"{TEST_FILES_DIR}/aluminium.lmp")

        assert self.data is not None

    def test_render_ovito(self) -> None:
        """Test the render_ovito method."""
        image = render_ovito(self.data.pipeline)
        assert isinstance(image, QImage)
        assert image.width() == 282
        assert image.height() == 262