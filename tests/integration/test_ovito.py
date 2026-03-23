"""Integration tests for the ovito backend — requires ovito to be installed."""

from __future__ import annotations

from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from unittest import TestCase
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
import pytest
from numpy.testing import assert_allclose

from agility.analysis import GBStructure, GBStructureTimeseries

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILES_DIR = MODULE_DIR.parent / "files"
SHEAR_DUMP_URL = "https://gitlab.com/ovito-org/ovito-sample-data/-/raw/master/tutorial/shear.dump"

# There is a breaking change in ovito 3.11 in the CNA modifier
if find_spec("ovito"):
    OVITO_VERSION = tuple(int(part) for part in version("ovito").split(".") if part.isdigit())
    BREAKING_VERSION = tuple(map(int, ["3", "11"]))
    BREAKING = OVITO_VERSION < BREAKING_VERSION


def _ensure_shear_dump() -> Path:
    """Ensure the OVITO shear.dump fixture is available locally."""
    filepath = TEST_FILES_DIR / "shear.dump"
    if filepath.exists():
        return filepath
    try:
        urlretrieve(SHEAR_DUMP_URL, filepath)  # noqa: S310
    except (OSError, URLError) as exc:
        pytest.skip(f"Unable to download shear.dump fixture: {exc}")
    return filepath


@pytest.mark.integration
@pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
class TestGBStructure(TestCase):
    """Test the GBStructure class with the ovito backend."""

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

    def test_grain_segmentation_orientations(self) -> None:
        """Test that grain orientations are stored after grain segmentation."""
        self.data.perform_ptm(enabled=("fcc"), output_orientation=True)
        orientations = self.data.get_distinct_grains()
        assert orientations is not None
        grain_count = self.data.pipeline.compute().attributes["GrainSegmentation.grain_count"]
        assert orientations.shape == (grain_count, 4)
        assert_allclose(np.linalg.norm(orientations, axis=1), np.ones(grain_count), atol=1e-6)


@pytest.mark.integration
@pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
class TestGBStructureOxide(TestCase):
    """Test the GBStructure class for an oxide structure with the ovito backend."""

    def setUp(self) -> None:
        """Set up the test."""
        self.data = GBStructure("ovito", f"{TEST_FILES_DIR}/STO_polycrystal.lmp")

        assert self.data is not None

    @pytest.mark.filterwarnings("ignore: Evaluating only the selected atoms. Be aware that")
    def test_expand_to_non_selected_(self) -> None:
        """Test the expansion to non-selected particles."""
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


@pytest.mark.integration
@pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
class TestGBStructureTimeseriesOvito(TestCase):
    """Integration tests for GBStructureTimeseries with ovito."""

    @pytest.mark.filterwarnings("ignore: Using all particles with a particle identifier as the")
    def test_gb_fraction_over_time(self) -> None:
        """Test grain-boundary fraction can be evaluated over all trajectory frames."""
        shear_dump = _ensure_shear_dump()
        ts = GBStructureTimeseries("ovito", shear_dump)

        assert ts.num_frames > 1
        ts.timestamps = list(range(ts.num_frames))
        assert ts.timestamps is not None
        assert len(ts.timestamps) == ts.num_frames

        ts.perform_cna(enabled=("fcc",), compute=False)
        gb_fractions: list[float] = []
        for frame_idx in range(ts.num_frames):
            ts.data = ts.pipeline.compute(frame=frame_idx)
            gb_fractions.append(ts.get_gb_fraction())

        assert ts.timestamps == list(range(ts.num_frames))
        assert len(gb_fractions) == ts.num_frames
        assert all(0.0 <= fraction <= 1.0 for fraction in gb_fractions)
        assert any(not np.isclose(gb_fractions[0], fraction) for fraction in gb_fractions[1:])
