"""Unit tests for analysis.py using mock objects — no real backends required."""

from __future__ import annotations

import types
from unittest import TestCase
from unittest.mock import MagicMock

import pytest

from agility.analysis import GBStructure, not_implemented


@pytest.mark.unit
class TestNotImplemented(TestCase):
    """Test the not_implemented helper function."""

    def test_returns_not_implemented_error(self) -> None:
        """Test that not_implemented returns a NotImplementedError instance."""
        err = not_implemented("babel")
        assert isinstance(err, NotImplementedError)

    def test_message_contains_backend_name(self) -> None:
        """Test that the error message includes the backend name."""
        backend = "pyiron"
        err = not_implemented(backend)
        assert backend in str(err)

    def test_can_be_raised(self) -> None:
        """Test that the returned error can be raised."""
        backend = "babel"
        with pytest.raises(NotImplementedError, match=backend):
            raise not_implemented(backend)


@pytest.mark.unit
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


@pytest.mark.unit
class TestClearSelectionUnit(TestCase):
    """Test _clear_selection using mock data objects — no backend packages required."""

    def _make_gbs(self, backend: str, selection: list[int]) -> GBStructure:
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = backend
        gbs.data = types.SimpleNamespace(selection=list(selection))
        return gbs

    def test_clear_selection_pymatgen(self) -> None:
        """Test that _clear_selection resets the selection list for the pymatgen backend."""
        gbs = self._make_gbs("pymatgen", [0, 1, 2])
        gbs._clear_selection()  # noqa: SLF001
        assert gbs.data.selection == []

    def test_clear_selection_ase(self) -> None:
        """Test that _clear_selection resets the selection list for the ase backend."""
        gbs = self._make_gbs("ase", [3, 5, 7])
        gbs._clear_selection()  # noqa: SLF001
        assert gbs.data.selection == []

    def test_clear_already_empty_selection(self) -> None:
        """Test that _clear_selection is a no-op when the selection is already empty."""
        gbs = self._make_gbs("pymatgen", [])
        gbs._clear_selection()  # noqa: SLF001
        assert gbs.data.selection == []


@pytest.mark.unit
class TestInvertSelectionUnit(TestCase):
    """Test _invert_selection using mock data objects — no backend packages required."""

    def _make_gbs(self, backend: str, selection: list[int], n_items: int) -> GBStructure:
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = backend
        gbs.data = types.SimpleNamespace(selection=list(selection))
        if backend == "pymatgen":
            gbs.data.structure = [None] * n_items
        else:
            gbs.data.atoms = [None] * n_items
        return gbs

    def test_invert_empty_selection_pymatgen(self) -> None:
        """Test that inverting an empty selection selects all sites (pymatgen backend)."""
        gbs = self._make_gbs("pymatgen", [], 4)
        gbs._invert_selection()  # noqa: SLF001
        assert gbs.data.selection == [0, 1, 2, 3]

    def test_invert_full_selection_ase(self) -> None:
        """Test that inverting a full selection yields an empty selection (ase backend)."""
        gbs = self._make_gbs("ase", [0, 1, 2, 3], 4)
        gbs._invert_selection()  # noqa: SLF001
        assert gbs.data.selection == []

    def test_invert_partial_selection(self) -> None:
        """Test that inverting a partial selection returns the complementary set."""
        gbs = self._make_gbs("pymatgen", [0, 2], 4)
        gbs._invert_selection()  # noqa: SLF001
        assert gbs.data.selection == [1, 3]

    def test_invert_twice_restores_original(self) -> None:
        """Test that inverting a selection twice restores the original selection."""
        original = [1, 3]
        gbs = self._make_gbs("ase", original, 4)
        gbs._invert_selection()  # noqa: SLF001
        gbs._invert_selection()  # noqa: SLF001
        assert gbs.data.selection == original
