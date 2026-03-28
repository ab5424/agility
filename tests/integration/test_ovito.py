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

    def test_get_tilt_angle(self) -> None:
        """Test tilt/twist decomposition on real grain orientations from aluminium.lmp.

        Runs PTM + grain segmentation on the aluminium polycrystal to obtain unit
        quaternion orientations, then decomposes the misorientation of every unique
        grain pair into tilt (Verkippungswinkel) and twist components relative to a
        [001] boundary plane normal.

        Expected values were computed from the grain orientations returned by
        ``GrainSegmentationModifier`` on ``aluminium.lmp``.
        """
        self.data.perform_ptm(enabled=("fcc"), output_orientation=True)
        orientations = self.data.get_distinct_grains()
        assert orientations is not None
        n_grains = len(orientations)
        assert n_grains == 6

        boundary_normal = np.array([0.0, 0.0, 1.0])

        # Expected (tilt_deg, twist_deg) for each unique grain pair (i, j),
        # relative to the [001] boundary normal.
        expected_pairwise: dict[tuple[int, int], tuple[float, float]] = {
            (0, 1): (48.10726876, 54.59075502),
            (0, 2): (21.62713430, 4.60089324),
            (0, 3): (50.71799925, 1.65595804),
            (0, 4): (27.35549143, 55.09193387),
            (0, 5): (63.96192738, 32.50641962),
            (1, 2): (69.95747423, 52.77113913),
            (1, 3): (42.72323838, 36.88942243),
            (1, 4): (44.25619581, 9.94485960),
            (1, 5): (30.07837598, 10.80567861),
            (2, 3): (68.72444263, 9.90248776),
            (2, 4): (42.36809186, 46.26167853),
            (2, 5): (84.35797406, 24.65154897),
            (3, 4): (33.37007453, 57.59625882),
            (3, 5): (24.50278740, 19.55583331),
            (4, 5): (49.80791529, 32.83230093),
        }

        for i in range(n_grains):
            for j in range(i + 1, n_grains):
                q_i = orientations[[i]]
                q_j = orientations[[j]]
                tilt, twist = self.data.get_tilt_angle(q_i, q_j, boundary_normal)

                assert tilt.shape == (1,)
                assert twist.shape == (1,)

                exp_tilt, exp_twist = expected_pairwise[(i, j)]
                assert_allclose(tilt[0], exp_tilt, atol=1e-4, err_msg=f"tilt ({i},{j})")
                assert_allclose(twist[0], exp_twist, atol=1e-4, err_msg=f"twist ({i},{j})")

        # Batch call: pass all consecutive pairs at once
        q_i_batch = orientations[:-1]
        q_j_batch = orientations[1:]
        tilt_batch, twist_batch = self.data.get_tilt_angle(q_i_batch, q_j_batch, boundary_normal)
        assert tilt_batch.shape == (n_grains - 1,)
        assert twist_batch.shape == (n_grains - 1,)
        assert_allclose(
            tilt_batch,
            [48.10726876, 69.95747423, 68.72444263, 33.37007453, 49.80791529],
            atol=1e-4,
        )
        assert_allclose(
            twist_batch,
            [54.59075502, 52.77113913, 9.90248776, 57.59625882, 32.83230093],
            atol=1e-4,
        )


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
            frame = ts.get_frame(frame_idx)
            gb_fractions.append(frame.get_gb_fraction())

        assert ts.timestamps == list(range(ts.num_frames))
        assert len(gb_fractions) == ts.num_frames
        assert all(0.0 <= fraction <= 1.0 for fraction in gb_fractions)
        assert any(not np.isclose(gb_fractions[0], fraction) for fraction in gb_fractions[1:])
        assert_allclose(
            [gb_fractions[0], gb_fractions[-1]],
            [0.1882845, 0.421025 if BREAKING else 0.2834728],
            rtol=1e-6,
        )
