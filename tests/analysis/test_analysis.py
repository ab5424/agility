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
    BREAKING_VERSION = tuple(map(int, "3.11".split(".")))
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
        from ovito.modifiers import GrainSegmentationModifier

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
