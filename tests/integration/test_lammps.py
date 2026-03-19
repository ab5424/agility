"""Integration tests for the LAMMPS backend — requires lammps to be installed."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from unittest import TestCase

import pytest

from agility.analysis import GBStructure

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILES_DIR = MODULE_DIR.parent / "files"


@pytest.mark.integration
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


@pytest.mark.integration
@pytest.mark.skipif(not find_spec("lammps"), reason="lammps not installed")
class TestGetTypeLammps(TestCase):
    """Test get_type for the lammps backend using the actual PyLAMMPS API."""

    def setUp(self) -> None:
        """Load a two-type LAMMPS data file for get_type tests."""
        self.gbs = GBStructure("lammps", f"{TEST_FILES_DIR}/test_two_types.lmp")

    def tearDown(self) -> None:
        """Close the LAMMPS instance after each test."""
        lmp = getattr(getattr(self, "gbs", None), "pylmp", None)
        if lmp is not None:
            lmp.lmp.close()

    def test_get_type_identifier_type1(self) -> None:
        """Test that get_type returns the correct atom IDs for type 1."""
        result = self.gbs.get_type(1)
        assert sorted(result) == [1, 3]

    def test_get_type_identifier_type2(self) -> None:
        """Test that get_type returns the correct atom IDs for type 2."""
        result = self.gbs.get_type(2)
        assert sorted(result) == [2, 4]

    def test_get_type_indices_type1(self) -> None:
        """Test that get_type returns correct array indices for type 1."""
        result = self.gbs.get_type(1, return_type="Indices")
        assert sorted(result) == [0, 2]

    def test_get_type_indices_type2(self) -> None:
        """Test that get_type returns correct array indices for type 2."""
        result = self.gbs.get_type(2, return_type="Indices")
        assert sorted(result) == [1, 3]

    def test_get_type_empty_for_nonexistent_type(self) -> None:
        """Test that get_type returns an empty list for a non-existent type."""
        result = self.gbs.get_type(99)
        assert result == []

    def test_get_type_invalid_return_type_raises(self) -> None:
        """Test that get_type raises NameError for an invalid return_type."""
        with pytest.raises(NameError, match="Only Indices and Identifier"):
            self.gbs.get_type(1, return_type="Invalid")


@pytest.mark.integration
@pytest.mark.skipif(not find_spec("lammps"), reason="lammps not installed")
class TestGetCrystallineAtomsLammps(TestCase):
    """Test get_crystalline_atoms for the lammps backend."""

    def setUp(self) -> None:
        """Set up a GBStructure with the lammps backend (no file loaded)."""
        self.gbs = GBStructure("lammps", "")

    def tearDown(self) -> None:
        """Close the LAMMPS instance after each test."""
        lmp = getattr(getattr(self, "gbs", None), "pylmp", None)
        if lmp is not None:
            lmp.lmp.close()

    def _setup_fcc_lattice(self) -> None:
        """Create a perfect FCC Al lattice for structural analysis tests."""
        lmp = self.gbs.pylmp
        lmp.units("metal")
        lmp.atom_style("atomic")
        lmp.lattice("fcc 4.05")
        lmp.region("box block 0 3 0 3 0 3")
        lmp.create_box("1 box")
        lmp.create_atoms("1 box")
        lmp.mass("1 26.982")
        lmp.pair_style("zero 6.0")
        lmp.pair_coeff("* *")
        lmp.run("0")

    def _setup_isolated_atoms(self) -> None:
        """Create two isolated atoms with no FCC-like neighbors (no crystalline structure)."""
        lmp = self.gbs.pylmp
        lmp.units("metal")
        lmp.atom_style("atomic")
        lmp.region("box block 0 20 0 20 0 20")
        lmp.create_box("1 box")
        # Place two atoms far apart so each has no neighbors within CNA cutoff
        lmp.create_atoms("1 single 5.0 5.0 5.0 units box")
        lmp.create_atoms("1 single 15.0 15.0 15.0 units box")
        lmp.mass("1 26.982")
        lmp.pair_style("zero 6.0")
        lmp.pair_coeff("* *")
        lmp.run("0")

    def test_invalid_mode_raises_value_error(self) -> None:
        """Test that an unrecognised mode raises ValueError."""
        with pytest.raises(ValueError, match="Incorrect mode"):
            self.gbs.get_crystalline_atoms(mode="invalid")

    def test_unimplemented_mode_raises_not_implemented(self) -> None:
        """Test that voronoi mode raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            self.gbs.get_crystalline_atoms(mode="voronoi")

    def test_invalid_return_type_raises_not_implemented(self) -> None:
        """Test that an invalid return_type raises NotImplementedError after CNA."""
        self._setup_fcc_lattice()
        self.gbs.perform_cna(cutoff=3.3)
        self.gbs.pylmp.run("0")
        with pytest.raises(NotImplementedError):
            self.gbs.get_crystalline_atoms(mode="cna", return_type="Invalid")

    def test_cna_perfect_fcc_all_crystalline_identifier(self) -> None:
        """All atoms in a perfect FCC lattice should be crystalline (CNA, Identifier)."""
        self._setup_fcc_lattice()
        self.gbs.perform_cna(cutoff=3.3)
        self.gbs.pylmp.run("0")
        n_atoms = self.gbs.pylmp.system.natoms
        result = self.gbs.get_crystalline_atoms(mode="cna")
        assert len(result) == n_atoms
        # LAMMPS create_atoms assigns sequential IDs starting from 1
        assert sorted(result) == list(range(1, n_atoms + 1))

    def test_cna_perfect_fcc_all_crystalline_indices(self) -> None:
        """All atoms in a perfect FCC lattice should be crystalline (CNA, Indices)."""
        self._setup_fcc_lattice()
        self.gbs.perform_cna(cutoff=3.3)
        self.gbs.pylmp.run("0")
        n_atoms = self.gbs.pylmp.system.natoms
        result = self.gbs.get_crystalline_atoms(mode="cna", return_type="Indices")
        assert len(result) == n_atoms
        assert sorted(result) == list(range(n_atoms))

    def test_cna_isolated_atoms_not_crystalline(self) -> None:
        """Atoms with no FCC-like neighbors (type 5 in CNA) must be excluded."""
        self._setup_isolated_atoms()
        self.gbs.perform_cna(cutoff=3.3)
        self.gbs.pylmp.run("0")
        # Two isolated atoms have no FCC-like neighbors → CNA assigns them type 5 ("other")
        result = self.gbs.get_crystalline_atoms(mode="cna")
        assert result == []
        # Verify the inverse: both isolated atoms appear as non-crystalline
        non_crystalline = self.gbs.get_non_crystalline_atoms(mode="cna")
        assert len(non_crystalline) == 2

    def test_cna_crystalline_and_non_crystalline_complement(self) -> None:
        """get_crystalline_atoms and get_non_crystalline_atoms must together cover all atoms."""
        self._setup_fcc_lattice()
        self.gbs.perform_cna(cutoff=3.3)
        self.gbs.pylmp.run("0")
        n_atoms = self.gbs.pylmp.system.natoms
        crystalline = self.gbs.get_crystalline_atoms(mode="cna")
        non_crystalline = self.gbs.get_non_crystalline_atoms(mode="cna")
        assert len(crystalline) + len(non_crystalline) == n_atoms
        assert set(crystalline).isdisjoint(set(non_crystalline))
