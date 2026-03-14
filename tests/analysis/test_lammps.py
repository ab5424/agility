"""Test LAMMPS backend and minimiser functionality."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

import pytest

from agility.analysis import GBStructure
from agility.minimiser import minimise_lmp

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILES_DIR = MODULE_DIR.parent / "files"


class TestMinimiseLmp(TestCase):
    """Test the minimise_lmp function using mock LAMMPS objects."""

    def test_wrong_min_opt_length_raises_value_error(self) -> None:
        """Test that a min_opt with wrong length raises ValueError."""
        mock_lmp = MagicMock()
        with pytest.raises(ValueError, match="four arguments"):
            minimise_lmp(mock_lmp, min_opt=(0, 1e-8, 1000))

    def test_minimise_lmp_calls_correct_methods(self) -> None:
        """Test that minimise_lmp calls the expected LAMMPS methods with default args."""
        mock_lmp = MagicMock()
        result = minimise_lmp(mock_lmp)
        mock_lmp.min_style.assert_called_once_with("fire")
        # The four default min_opt values are formatted into a single space-separated string
        default_min_opt = (0, 1e-8, 1000, 100000)
        mock_lmp.minimize.assert_called_once_with(
            f"{default_min_opt[0]} {default_min_opt[1]} {default_min_opt[2]} {default_min_opt[3]}",
        )
        assert result is mock_lmp

    def test_minimise_lmp_custom_style(self) -> None:
        """Test that minimise_lmp forwards the requested minimization style."""
        mock_lmp = MagicMock()
        minimise_lmp(mock_lmp, style="cg")
        mock_lmp.min_style.assert_called_once_with("cg")

    def test_minimise_lmp_with_mod(self) -> None:
        """Test that minimise_lmp applies min_modify commands when mod is given."""
        mock_lmp = MagicMock()
        minimise_lmp(mock_lmp, mod=[("line", "quadratic")])
        mock_lmp.min_modify.assert_called_once_with("line quadratic")

    def test_minimise_lmp_returns_lmp(self) -> None:
        """Test that minimise_lmp returns the LAMMPS object."""
        mock_lmp = MagicMock()
        result = minimise_lmp(mock_lmp)
        assert result is mock_lmp


@pytest.mark.skipif(not find_spec("lammps"), reason="lammps not installed")
class TestGBStructureLammps(TestCase):
    """Test the GBStructure class with the LAMMPS backend."""

    def setUp(self) -> None:
        """Set up a GBStructure with the lammps backend (no file loaded)."""
        self.gbs = GBStructure("lammps", "")

    def tearDown(self) -> None:
        """Close the LAMMPS instance after each test."""
        lmp = getattr(getattr(self, "gbs", None), "pylmp", None)
        if lmp is not None:
            lmp.lmp.close()

    def test_backend_attribute(self) -> None:
        """Test that the backend attribute is set correctly."""
        assert self.gbs.backend == "lammps"

    def test_init_creates_pylmp(self) -> None:
        """Test that initializing with the lammps backend creates a PyLammps instance."""
        from lammps import PyLammps  # noqa: PLC0415

        assert isinstance(self.gbs.pylmp, PyLammps)

    def test_invalid_file_type_raises_value_error(self) -> None:
        """Test that _init_lmp raises ValueError for an unrecognised file type."""
        with pytest.raises(ValueError, match="type of lammps file"):
            self.gbs._init_lmp("nonexistent.lmp", file_type="invalid")  # noqa: SLF001

    def test_read_lammps_data_file(self) -> None:
        """Test reading a charge-style LAMMPS data file via GBStructure."""
        gbs = GBStructure("lammps", f"{TEST_FILES_DIR}/test_charge.lmp")
        try:
            assert gbs.pylmp.system.natoms == 2
        finally:
            gbs.pylmp.lmp.close()
