"""Integration tests for polycrystal.py — requires atomsk to be installed."""

from __future__ import annotations

import pathlib
import tempfile
from pathlib import Path
from unittest import TestCase

import pytest

from agility.polycrystal import PolycrystalBuilder, find_atomsk

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILES_DIR = MODULE_DIR.parent / "files"

# Evaluate once at collection time so the skip decorator works correctly.
ATOMSK_PATH = find_atomsk()


@pytest.mark.integration
@pytest.mark.skipif(ATOMSK_PATH is None, reason="atomsk not installed")
class TestFindAtomskIntegration(TestCase):
    """Test find_atomsk() with a real atomsk installation."""

    def test_find_atomsk_returns_executable(self) -> None:
        """Test that find_atomsk returns a path pointing to an actual file."""
        path = find_atomsk()
        assert path is not None
        assert pathlib.Path(path).is_file()

    def test_find_atomsk_is_idempotent(self) -> None:
        """Test that successive calls to find_atomsk return the same path."""
        assert find_atomsk() == find_atomsk()


@pytest.mark.integration
@pytest.mark.skipif(ATOMSK_PATH is None, reason="atomsk not installed")
class TestPolycrystalBuilderIntegration(TestCase):
    """End-to-end tests for PolycrystalBuilder using a real atomsk binary."""

    def setUp(self) -> None:
        """Set up a PolycrystalBuilder pointing at the FCC Al unit cell fixture."""
        self.unit_cell = TEST_FILES_DIR / "Al_fcc.vasp"
        self.builder = PolycrystalBuilder(self.unit_cell)

    def test_build_explicit_grains_creates_output_file(self) -> None:
        """Test that build() with explicit grain seeds produces the output file."""
        self.builder.set_box(50.0, 50.0, 50.0)
        self.builder.add_grain((12.5, 25.0, 25.0), (0.0, 0.0, 0.0))
        self.builder.add_grain((37.5, 25.0, 25.0), (45.0, 0.0, 0.0))
        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "poly.lmp"
            result = self.builder.build(output, output_format="lmp")
            assert result.exists()
            assert result.stat().st_size > 0

    def test_build_random_grains_creates_output_file(self) -> None:
        """Test that build() with randomly placed grains produces the output file."""
        self.builder.set_box(50.0, 50.0, 50.0)
        self.builder.set_random_grains(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "poly_random.lmp"
            result = self.builder.build(output, output_format="lmp")
            assert result.exists()
            assert result.stat().st_size > 0

    def test_build_returns_correct_path(self) -> None:
        """Test that build() returns a pathlib.Path equal to the requested output path."""
        self.builder.set_box(40.0, 40.0, 40.0)
        self.builder.set_random_grains(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "poly.lmp"
            result = self.builder.build(output, output_format="lmp")
            assert isinstance(result, pathlib.Path)
            assert result == output.resolve()

    def test_build_without_output_format_returns_requested_path(self) -> None:
        """Test that build() returns the requested resolved path when output_format is omitted."""
        self.builder.set_box(40.0, 40.0, 40.0)
        self.builder.set_random_grains(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "poly.lmp"
            result = self.builder.build(output)
            assert isinstance(result, pathlib.Path)
            assert result == output.resolve()
            assert result.exists()

    def test_build_vasp_output_format(self) -> None:
        """Test that build() can write a VASP POSCAR output file."""
        self.builder.set_box(40.0, 40.0, 40.0)
        self.builder.set_random_grains(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "poly.vasp"
            result = self.builder.build(output, output_format="vasp")
            assert result.exists()
            assert result.stat().st_size > 0

    def test_builder_accepts_string_unit_cell_path(self) -> None:
        """Test that PolycrystalBuilder works when the unit cell is given as a string."""
        builder = PolycrystalBuilder(str(self.unit_cell))
        builder.set_box(40.0, 40.0, 40.0)
        builder.set_random_grains(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "poly.lmp"
            result = builder.build(output, output_format="lmp")
            assert result.exists()

    def test_build_with_extra_options(self) -> None:
        """Test that extra_options flags are forwarded to atomsk without error."""
        self.builder.set_box(40.0, 40.0, 40.0)
        self.builder.set_random_grains(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "poly.lmp"
            # -overwrite suppresses the "file already exists" interactive prompt
            result = self.builder.build(output, output_format="lmp", extra_options=["-overwrite"])
            assert result.exists()

    def test_build_two_grains_lammps_file_has_atoms(self) -> None:
        """Test that the LAMMPS output file produced for 2 explicit grains contains atoms."""
        self.builder.set_box(50.0, 50.0, 50.0)
        self.builder.add_grain((12.5, 25.0, 25.0), (0.0, 0.0, 0.0))
        self.builder.add_grain((37.5, 25.0, 25.0), (45.0, 0.0, 0.0))
        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "poly.lmp"
            self.builder.build(output, output_format="lmp")
            content = output.read_text(encoding="utf-8")
            # A valid LAMMPS data file must declare the atom count
            assert "atoms" in content
