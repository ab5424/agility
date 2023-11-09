"""Test the Crystal class."""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase

from agility.analysis import GBStructure

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILES_DIR = Path(MODULE_DIR / ".." / ".." / "tests" / "files")


class TestGBStructure(TestCase):
    """Test the Crystal class."""

    def setUp(self) -> None:
        """Set up the test."""
        self.data = GBStructure("ovito", f"{TEST_FILES_DIR}/aluminium.lmp")

        assert self.data is not None

    def test_cna(self) -> None:
        """Test the get_data method."""
        self.data.perform_cna(enabled=("fcc"))
        crystalline_atoms = self.data.get_crystalline_atoms()
        non_crystalline_atoms = self.data.get_non_crystalline_atoms()
        assert len(crystalline_atoms) == 4320
        assert len(non_crystalline_atoms) == 3361
