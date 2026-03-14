"""Test the analysis functions."""

from __future__ import annotations

from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest
from numpy.testing import assert_allclose

from agility.analysis import GBStructure

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILES_DIR = Path(MODULE_DIR / ".." / ".." / "tests" / "files")

# There is a breaking change in ovito 3.11 in the CNA modifier
if find_spec("ovito"):
    OVITO_VERSION = tuple(int(part) for part in version("ovito").split(".") if part.isdigit())
    BREAKING_VERSION = tuple(map(int, ["3", "11"]))
    BREAKING = OVITO_VERSION < BREAKING_VERSION


@pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
class TestGBStructure(TestCase):
    """Test the GBStructure class."""

    def setUp(self) -> None:
        """Set up the test."""
        self.data = GBStructure("ovito", f"{TEST_FILES_DIR}/aluminium.lmp")

        assert self.data is not None

    def test_cna(self) -> None:
        """Test Common Neighbor Analysis method."""
        self.data.perform_cna(enabled=("fcc"))
        crystalline_atoms = self.data.get_crystalline_atoms()
        non_crystalline_atoms = self.data.get_non_crystalline_atoms()
        assert len(crystalline_atoms) == (4320 if BREAKING else 4330)
        assert len(non_crystalline_atoms) == (3361 if BREAKING else 3351)
        self.data.perform_cna(mode="AdaptiveCutoff", enabled=("fcc"))
        crystalline_atoms = self.data.get_crystalline_atoms()
        non_crystalline_atoms = self.data.get_non_crystalline_atoms()
        assert len(crystalline_atoms) == (4275 if BREAKING else 4277)
        assert len(non_crystalline_atoms) == (3406 if BREAKING else 3404)

    def test_ptm(self) -> None:
        """Test Polyhedral Template Matching method."""
        self.data.perform_ptm(enabled=("fcc"))
        crystalline_atoms = self.data.get_crystalline_atoms()
        non_crystalline_atoms = self.data.get_non_crystalline_atoms()
        assert len(crystalline_atoms) == 4390
        assert len(non_crystalline_atoms) == 3291

    @pytest.mark.filterwarnings("ignore: Using all particles with a particle identifier as the")
    def test_gb_fraction(self) -> None:
        """Test the GB fraction method."""
        self.data.perform_cna(enabled=("fcc"))
        gb_fraction = self.data.get_gb_fraction()
        assert_allclose(gb_fraction, float(3361 / 7681 if BREAKING else 3351 / 7681))

    def test_grain_segmentation(self) -> None:
        """Test the grain segmentation method."""
        from ovito.modifiers import GrainSegmentationModifier  # noqa: PLC0415

        self.data.perform_ptm(enabled=("fcc"), output_orientation=True)
        self.data.get_distinct_grains(compute=False)
        assert isinstance(self.data.pipeline.modifiers[1], GrainSegmentationModifier)
        assert self.data.pipeline.compute().attributes["GrainSegmentation.grain_count"] == 6


@pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
class TestGBStructureOxide(TestCase):
    """Test the GBStructure class for an oxide."""

    def setUp(self) -> None:
        """Set up the test."""
        self.data = GBStructure("ovito", f"{TEST_FILES_DIR}/STO_polycrystal.lmp")

        assert self.data is not None

    @pytest.mark.filterwarnings("ignore: Evaluating only the selected atoms. Be aware that")
    def test_expand_to_non_selected_(self) -> None:
        """Test the expansion to non-selected ."""
        self.data.set_analysis()
        sr = self.data.get_type(3)
        ti = self.data.get_type(2)
        o = self.data.get_type(1)

        self.data.select_particles_by_type({"Sr", "Ti"})
        self.data.set_analysis()
        selected_particles = set(np.where(self.data.data.particles.selection == 1)[0])
        assert len(selected_particles) == len(sr) + len(ti)

        self.data.perform_cna(enabled=("bcc"), only_selected=True)
        assert len(self.data.get_crystalline_atoms()) == 2562
        assert len(self.data.get_non_crystalline_atoms()) == self.data.data.particles.count - 2562

        non_cryst_anions = self.data.expand_to_non_selected(nearest_n=12)
        assert len(non_cryst_anions) == 2582
        assert len(o) - len(non_cryst_anions) == 3748


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


@pytest.mark.skipif(not find_spec("ase"), reason="ase not installed")
class TestGBStructureASE(TestCase):
    """Test the GBStructure class with the ASE backend."""

    def setUp(self) -> None:
        """Set up the test."""
        self.gbs = GBStructure("ase", f"{TEST_FILES_DIR}/NaCl.vasp")

        assert self.gbs is not None

    def test_read_file(self) -> None:
        """Test that the structure is loaded correctly from file."""
        assert self.gbs.data is not None
        assert self.gbs.data.atoms is not None
        # NaCl has 4 Na + 4 Cl = 8 atoms
        assert len(self.gbs.data.atoms) == 8

    def test_read_file_selection_initialized(self) -> None:
        """Test that the selection list is initialized empty after reading a file."""
        assert self.gbs.data.selection == []

    def test_delete_particles(self) -> None:
        """Test that delete_particles removes the specified species."""
        self.gbs.delete_particles(["Na"])
        # Only 4 Cl atoms should remain
        assert len(self.gbs.data.atoms) == 4
        symbols_remaining = set(self.gbs.data.atoms.get_chemical_symbols())
        assert symbols_remaining == {"Cl"}

    def test_delete_particles_resets_selection(self) -> None:
        """Test that delete_particles resets the selection list after mutating the structure."""
        self.gbs.data.selection = [4, 5, 6, 7]
        self.gbs.delete_particles(["Na"])
        assert self.gbs.data.selection == []

    def test_invert_selection_empty(self) -> None:
        """Test that inverting an empty selection selects all atoms."""
        n_atoms = len(self.gbs.data.atoms)
        assert self.gbs.data.selection == []

        self.gbs._invert_selection()  # noqa: SLF001

        assert self.gbs.data.selection == list(range(n_atoms))

    def test_invert_selection_full(self) -> None:
        """Test that inverting a full selection yields an empty selection."""
        n_atoms = len(self.gbs.data.atoms)
        self.gbs.data.selection = list(range(n_atoms))

        self.gbs._invert_selection()  # noqa: SLF001

        assert self.gbs.data.selection == []

    def test_invert_selection_partial(self) -> None:
        """Test that inverting a partial selection returns the complementary set."""
        n_atoms = len(self.gbs.data.atoms)
        # Select the first 3 atoms
        self.gbs.data.selection = [0, 1, 2]

        self.gbs._invert_selection()  # noqa: SLF001

        assert self.gbs.data.selection == list(range(3, n_atoms))

    def test_invert_selection_twice(self) -> None:
        """Test that inverting a selection twice restores the original selection."""
        original_selection = [0, 2, 5]
        self.gbs.data.selection = list(original_selection)

        self.gbs._invert_selection()  # noqa: SLF001
        self.gbs._invert_selection()  # noqa: SLF001

        assert self.gbs.data.selection == original_selection

    def test_delete_selection(self) -> None:
        """Test that _delete_selection removes the selected atoms from the structure."""
        # Select the 4 Na atoms (indices 0-3 in the NaCl structure)
        self.gbs.data.selection = [0, 1, 2, 3]

        self.gbs._delete_selection()  # noqa: SLF001

        # 4 atoms should have been removed, leaving 4 Cl atoms
        assert len(self.gbs.data.atoms) == 4
        assert self.gbs.data.selection == []

    def test_invert_then_delete_selection(self) -> None:
        """Test the invert+delete workflow removes the non-selected atoms."""
        n_atoms = len(self.gbs.data.atoms)
        # Start with 4 Na atoms selected
        self.gbs.data.selection = [0, 1, 2, 3]

        # Invert: now the 4 Cl atoms (indices 4-7) are selected
        self.gbs._invert_selection()  # noqa: SLF001
        assert self.gbs.data.selection == list(range(4, n_atoms))

        # Delete the selected (Cl) atoms
        self.gbs._delete_selection()  # noqa: SLF001

        # Only the 4 Na atoms should remain
        assert len(self.gbs.data.atoms) == 4
        symbols_remaining = set(self.gbs.data.atoms.get_chemical_symbols())
        assert symbols_remaining == {"Na"}
        assert self.gbs.data.selection == []

    def test_select_particles_invert_delete(self) -> None:
        """Test select_particles via public API: select Na, invert, delete → only Na remains."""
        # Select the 4 Na atoms (indices 0-3), invert (selects Cl), then delete Cl
        self.gbs.select_particles([0, 1, 2, 3], invert=True, delete=True)

        # Only the 4 Na atoms should remain
        assert len(self.gbs.data.atoms) == 4
        symbols_remaining = set(self.gbs.data.atoms.get_chemical_symbols())
        assert symbols_remaining == {"Na"}
        assert self.gbs.data.selection == []

    def test_select_particles_no_invert_no_delete(self) -> None:
        """Test select_particles sets the selection without inverting or deleting."""
        self.gbs.select_particles([2, 3, 4], invert=False, delete=False)

        assert self.gbs.data.selection == [2, 3, 4]
        # Structure is unchanged
        assert len(self.gbs.data.atoms) == 8
