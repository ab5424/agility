"""Integration tests for the pymatgen backend — requires pymatgen to be installed."""

from __future__ import annotations

import tempfile
from importlib.util import find_spec
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

import pytest

from agility.analysis import GBStructure

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILES_DIR = MODULE_DIR.parent / "files"


@pytest.mark.integration
@pytest.mark.skipif(not find_spec("pymatgen"), reason="pymatgen not installed")
class TestGBStructurePymatgen(TestCase):
    """Test the GBStructure class with the pymatgen backend."""

    def setUp(self) -> None:
        """Set up the test."""
        self.gbs = GBStructure("pymatgen", f"{TEST_FILES_DIR}/NaCl.vasp")

        assert self.gbs is not None

    def test_read_file(self) -> None:
        """Test that the structure is loaded correctly from file."""
        assert self.gbs.data is not None
        assert self.gbs.data.structure is not None
        # NaCl has 4 Na + 4 Cl = 8 sites
        assert len(self.gbs.data.structure) == 8

    def test_read_file_selection_initialized(self) -> None:
        """Test that the selection list is initialized empty after reading a file."""
        assert self.gbs.data.selection == []

    def test_delete_particles(self) -> None:
        """Test that delete_particles removes the specified species."""
        self.gbs.delete_particles(["Na"])
        # Only 4 Cl atoms should remain
        assert len(self.gbs.data.structure) == 4
        species_remaining = {str(s) for s in self.gbs.data.structure.species}
        assert species_remaining == {"Cl"}

    def test_delete_particles_resets_selection(self) -> None:
        """Test that delete_particles resets the selection list after mutating the structure."""
        self.gbs.data.selection = [4, 5, 6, 7]
        self.gbs.delete_particles(["Na"])
        assert self.gbs.data.selection == []

    def test_invert_selection_empty(self) -> None:
        """Test that inverting an empty selection selects all sites."""
        n_sites = len(self.gbs.data.structure)
        assert self.gbs.data.selection == []

        self.gbs._invert_selection()  # noqa: SLF001

        assert self.gbs.data.selection == list(range(n_sites))

    def test_invert_selection_full(self) -> None:
        """Test that inverting a full selection yields an empty selection."""
        n_sites = len(self.gbs.data.structure)
        self.gbs.data.selection = list(range(n_sites))

        self.gbs._invert_selection()  # noqa: SLF001

        assert self.gbs.data.selection == []

    def test_invert_selection_partial(self) -> None:
        """Test that inverting a partial selection returns the complementary set."""
        n_sites = len(self.gbs.data.structure)
        # Select the first 3 sites
        self.gbs.data.selection = [0, 1, 2]

        self.gbs._invert_selection()  # noqa: SLF001

        assert self.gbs.data.selection == list(range(3, n_sites))

    def test_invert_selection_twice(self) -> None:
        """Test that inverting a selection twice restores the original selection."""
        original_selection = [0, 2, 5]
        self.gbs.data.selection = list(original_selection)

        self.gbs._invert_selection()  # noqa: SLF001
        self.gbs._invert_selection()  # noqa: SLF001

        assert self.gbs.data.selection == original_selection

    def test_delete_selection(self) -> None:
        """Test that _delete_selection removes the selected sites from the structure."""
        # Select the 4 Na sites (indices 0-3 in the NaCl structure)
        self.gbs.data.selection = [0, 1, 2, 3]

        self.gbs._delete_selection()  # noqa: SLF001

        # 4 sites should have been removed, leaving 4 Cl sites
        assert len(self.gbs.data.structure) == 4
        assert self.gbs.data.selection == []

    def test_invert_then_delete_selection(self) -> None:
        """Test the invert+delete workflow removes the non-selected sites."""
        n_sites = len(self.gbs.data.structure)
        # Start with 4 Na sites selected
        self.gbs.data.selection = [0, 1, 2, 3]

        # Invert: now the 4 Cl sites (indices 4-7) are selected
        self.gbs._invert_selection()  # noqa: SLF001
        assert self.gbs.data.selection == list(range(4, n_sites))

        # Delete the selected (Cl) sites
        self.gbs._delete_selection()  # noqa: SLF001

        # Only the 4 Na sites should remain
        assert len(self.gbs.data.structure) == 4
        species_remaining = {str(s) for s in self.gbs.data.structure.species}
        assert species_remaining == {"Na"}
        assert self.gbs.data.selection == []

    def test_select_particles_invert_delete(self) -> None:
        """Test select_particles via public API: select Na, invert, delete → only Na remains."""
        # Select the 4 Na sites (indices 0-3), invert (selects Cl), then delete Cl
        self.gbs.select_particles([0, 1, 2, 3], invert=True, delete=True)

        # Only the 4 Na sites should remain
        assert len(self.gbs.data.structure) == 4
        species_remaining = {str(s) for s in self.gbs.data.structure.species}
        assert species_remaining == {"Na"}
        assert self.gbs.data.selection == []

    def test_select_particles_no_invert_no_delete(self) -> None:
        """Test select_particles sets the selection without inverting or deleting."""
        self.gbs.select_particles([2, 3, 4], invert=False, delete=False)

        assert self.gbs.data.selection == [2, 3, 4]
        # Structure is unchanged
        assert len(self.gbs.data.structure) == 8

    def test_minimise(self) -> None:
        """Test that minimise calls Structure.relax() and updates the structure."""
        mock_calculator = MagicMock()
        mock_relaxed_structure = MagicMock()
        original_structure = self.gbs.data.structure
        original_structure.relax = MagicMock(return_value=mock_relaxed_structure)

        self.gbs.minimise(calculator=mock_calculator)

        original_structure.relax.assert_called_once_with(calculator=mock_calculator)
        assert self.gbs.data.structure is mock_relaxed_structure

    def test_minimise_with_trajectory(self) -> None:
        """Test that minimise correctly unwraps tuple return from Structure.relax()."""
        mock_calculator = MagicMock()
        mock_relaxed_structure = MagicMock()
        mock_trajectory = MagicMock()
        original_structure = self.gbs.data.structure
        original_structure.relax = MagicMock(return_value=(mock_relaxed_structure, mock_trajectory))

        self.gbs.minimise(calculator=mock_calculator, return_trajectory=True)

        original_structure.relax.assert_called_once_with(
            calculator=mock_calculator,
            return_trajectory=True,
        )
        assert self.gbs.data.structure is mock_relaxed_structure

    def test_save_structure_cif(self) -> None:
        """Test that save_structure writes a CIF file with the pymatgen backend."""
        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
            tmp_path = tmp.name
        self.addCleanup(lambda: Path(tmp_path).unlink(missing_ok=True))
        self.gbs.save_structure(tmp_path, "cif")
        assert Path(tmp_path).stat().st_size > 0

    def test_save_structure_poscar(self) -> None:
        """Test that save_structure writes a POSCAR file with the pymatgen backend."""
        with tempfile.NamedTemporaryFile(suffix=".vasp", delete=False) as tmp:
            tmp_path = tmp.name
        self.addCleanup(lambda: Path(tmp_path).unlink(missing_ok=True))
        self.gbs.save_structure(tmp_path, "poscar")
        assert Path(tmp_path).stat().st_size > 0
