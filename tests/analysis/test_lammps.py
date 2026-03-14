"""Test LAMMPS backend and minimiser functionality."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import pytest

from agility.analysis import GBStructure
from agility.minimiser import minimise_lmp

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILES_DIR = MODULE_DIR.parent / "files"


class TestSaveStructureLammps(TestCase):
    """Test save_structure for the lammps backend using a mock LAMMPS object."""

    def setUp(self) -> None:
        """Set up a GBStructure with a mocked pylmp."""
        self.gbs = GBStructure.__new__(GBStructure)
        self.gbs.backend = "lammps"
        self.gbs.pylmp = MagicMock()

    def test_save_structure_invalid_file_type_raises_value_error(self) -> None:
        """Test that save_structure raises ValueError for an unknown file type."""
        with pytest.raises(ValueError, match="Unrecognised file type"):
            self.gbs.save_structure("out.xyz", "xyz")

    def test_save_structure_data_delegates_to_pylmp(self) -> None:
        """Test that save_structure calls write_data for file_type='data'."""
        self.gbs.save_structure("out.lmp", "data")
        self.gbs.pylmp.write_data.assert_called_once_with("out.lmp")

    def test_save_structure_dump_delegates_to_pylmp(self) -> None:
        """Test that save_structure calls write_dump for file_type='dump'."""
        self.gbs.save_structure("out.dump", "dump")
        self.gbs.pylmp.write_dump.assert_called_once_with("out.dump")

    def test_save_structure_restart_delegates_to_pylmp(self) -> None:
        """Test that save_structure calls write_restart for file_type='restart'."""
        self.gbs.save_structure("out.restart", "restart")
        self.gbs.pylmp.write_restart.assert_called_once_with("out.restart")


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


class TestGetTypeLammps(TestCase):
    """Test get_type for the lammps backend using a mock LAMMPS object."""

    def setUp(self) -> None:
        """Set up a GBStructure with a mocked pylmp and 5 atoms of 3 types."""
        self.gbs = GBStructure.__new__(GBStructure)
        self.gbs.backend = "lammps"
        self.gbs.pylmp = MagicMock()
        # Atom IDs: [1, 2, 3, 4, 5], types: [1, 2, 1, 3, 2]
        self.ids = np.array([[1], [2], [3], [4], [5]])
        self.atom_types = np.array([[1], [2], [1], [3], [2]])
        self.gbs.pylmp.lmp.numpy.extract_atom.side_effect = lambda name: (
            self.ids if name == "id" else self.atom_types
        )

    def test_get_type_identifier(self) -> None:
        """Test that get_type returns correct atom IDs for a given type."""
        result = self.gbs.get_type(1)
        assert sorted(result) == [1, 3]

    def test_get_type_identifier_type2(self) -> None:
        """Test that get_type returns correct atom IDs for type 2."""
        result = self.gbs.get_type(2)
        assert sorted(result) == [2, 5]

    def test_get_type_indices(self) -> None:
        """Test that get_type returns correct indices for a given type."""
        result = self.gbs.get_type(1, return_type="Indices")
        assert sorted(result) == [0, 2]

    def test_get_type_indices_type3(self) -> None:
        """Test that get_type returns correct indices for type 3."""
        result = self.gbs.get_type(3, return_type="Indices")
        assert sorted(result) == [3]

    def test_get_type_empty_for_nonexistent_type(self) -> None:
        """Test that get_type returns an empty list for a non-existent type."""
        result = self.gbs.get_type(99)
        assert list(result) == []

    def test_get_type_invalid_return_type_raises(self) -> None:
        """Test that get_type raises NameError for an invalid return_type."""
        with pytest.raises(NameError, match="Only Indices and Identifier"):
            self.gbs.get_type(1, return_type="Invalid")
