"""Integration tests for the ovito backend — requires ovito to be installed."""

from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path
from unittest import TestCase
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
import pytest
from numpy.testing import assert_allclose

from agility.analysis import GBStructure, GBStructureTimeseries

PYTHON_VERSION = sys.version_info
SKIP_OVITO = PYTHON_VERSION <= (3, 12)

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILES_DIR = MODULE_DIR.parent / "files"
SHEAR_DUMP_URL = "https://gitlab.com/ovito-org/ovito-sample-data/-/raw/master/tutorial/shear.dump"


def _ensure_shear_dump() -> Path:
    """Ensure the OVITO shear.dump fixture is available locally."""
    filepath = TEST_FILES_DIR / "shear.dump"
    if filepath.exists():
        return filepath
    try:
        urlretrieve(SHEAR_DUMP_URL, filepath)
    except (OSError, URLError) as exc:
        pytest.skip(f"Unable to download shear.dump fixture: {exc}")
    return filepath


@pytest.mark.integration
@pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
@pytest.mark.skipif(SKIP_OVITO, reason="Python <= 3.12 not supported for ovito")
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
        assert len(crystalline_atoms) == 4330
        assert len(non_crystalline_atoms) == 3351
        self.data.perform_cna(mode="AdaptiveCutoff", enabled=("fcc"))
        crystalline_atoms = self.data.get_crystalline_atoms()
        non_crystalline_atoms = self.data.get_non_crystalline_atoms()
        assert len(crystalline_atoms) == 4277
        assert len(non_crystalline_atoms) == 3404

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

        assert_allclose(gb_fraction, float(3351 / 7681))

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
        grain pair into tilt and twist components relative to a
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
@pytest.mark.skipif(SKIP_OVITO, reason="Python <= 3.12 not supported for ovito")
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
@pytest.mark.skipif(SKIP_OVITO, reason="Python <= 3.12 not supported for ovito")
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
            [0.1882845, 0.2834728],
            rtol=1e-6,
        )

    def test_calculate_and_get_displacements(self) -> None:
        """Test displacement calculation and retrieval across frames."""
        shear_dump = _ensure_shear_dump()
        ts = GBStructureTimeseries("ovito", shear_dump)

        ts.calculate_displacements(reference_frame=0)
        disp_0 = ts.get_displacements(frame_idx=0)
        assert disp_0.shape == (1912, 3)
        assert_allclose(disp_0, 0.0)

        disp_5 = ts.get_displacements(frame_idx=5)
        assert disp_5.shape == (1912, 3)
        assert np.max(np.linalg.norm(disp_5, axis=1)) > 0.0

    def test_get_time_array_from_trajectory_and_dt(self) -> None:
        """Test time array extraction from trajectory Timestep attribute and dt overwrite."""
        shear_dump = _ensure_shear_dump()
        ts = GBStructureTimeseries("ovito", shear_dump)

        # 1. Read from trajectory attribute 'Timestep'
        times = ts.get_time_array()
        assert len(times) == ts.num_frames
        assert times[0] == 0.0
        assert times[1] == 2000.0

        # 2. Overwrite with dt
        with pytest.warns(UserWarning, match="overwrites the timestep property"):
            times_dt = ts.get_time_array(dt=0.002)
        assert_allclose(times_dt[:3], [0.0, 0.002, 0.004])

    def test_msd_and_regions(self) -> None:
        """Test MSD calculation across regions and selection modes."""
        shear_dump = _ensure_shear_dump()
        ts = GBStructureTimeseries("ovito", shear_dump)
        ts.perform_cna(enabled=("fcc",), compute=False)

        # 1. All particles
        msd_all = ts.get_msd(region="all")
        assert len(msd_all) == ts.num_frames
        assert msd_all[0] == 0.0
        assert np.all(msd_all >= 0.0)
        assert msd_all[-1] > 0.0

        # 2. Grain boundary (initial residence)
        msd_gb_init = ts.get_msd(region="gb", selection_mode="initial")
        assert msd_gb_init[0] == 0.0
        assert np.all(msd_gb_init >= 0.0)

        # 3. Grain boundary (continuous residence)
        msd_gb_cont = ts.get_msd(region="gb", selection_mode="continuous")
        assert msd_gb_cont[0] == 0.0
        assert np.all(msd_gb_cont >= 0.0)

        # 4. Bulk and grain edge
        msd_bulk = ts.get_msd(region="bulk")
        msd_edge = ts.get_msd(region="grain_edge")
        assert msd_bulk[0] == 0.0
        assert msd_edge[0] == 0.0

    def test_region_residence(self) -> None:
        """Test tracking particle residence in grain boundary over time."""
        shear_dump = _ensure_shear_dump()
        ts = GBStructureTimeseries("ovito", shear_dump)
        ts.perform_cna(enabled=("fcc",), compute=False)

        res = ts.get_region_residence(region="gb")
        assert len(res["initial_ids"]) == 360
        assert len(res["counts"]) == ts.num_frames
        assert len(res["fraction_remaining"]) == ts.num_frames
        assert res["fraction_remaining"][0] == 1.0
        assert all(0.0 <= f <= 1.0 for f in res["fraction_remaining"])

    def test_get_diffusion_coefficient(self) -> None:
        """Test extraction of diffusion coefficient D with fit statistics and units."""
        shear_dump = _ensure_shear_dump()
        ts = GBStructureTimeseries("ovito", shear_dump)

        # Fit with trajectory timesteps (where dt is in timesteps/steps, D in Angstrom^2 / step)
        d_coeff, fit = ts.get_diffusion_coefficient(return_fit=True)
        assert d_coeff > 0.0
        assert "slope" in fit
        assert "rvalue" in fit
        assert fit["slope"] > 0.0

        # Fit with explicit dt in picoseconds [ps], e.g. 2000 steps * 1 fs = 2.0 ps
        dt_ps = 2.0
        with pytest.warns(UserWarning, match="overwrites the timestep property"):
            d_coeff_ps = ts.get_diffusion_coefficient(dt=dt_ps)
        assert d_coeff_ps > 0.0

        # Unit conversion: 1 Angstrom^2 / ps = 1e-4 cm^2 / s = 1e-8 m^2 / s
        d_coeff_m2_s = d_coeff_ps * 1e-8
        d_coeff_cm2_s = d_coeff_ps * 1e-4
        assert_allclose(d_coeff_m2_s * 1e4, d_coeff_cm2_s)

        # Provide dt in seconds [s] directly (2.0 ps = 2.0e-12 s) -> D in Angstrom^2 / s
        dt_s = dt_ps * 1e-12
        with pytest.warns(UserWarning, match="overwrites the timestep property"):
            d_coeff_ang2_s = ts.get_diffusion_coefficient(dt=dt_s)
        # Convert Angstrom^2 / s to m^2 / s (1 Angstrom^2 = 1e-20 m^2)
        assert_allclose(d_coeff_ang2_s * 1e-20, d_coeff_m2_s)

    def test_diffusion_crystalline_and_boundary_comparison(self) -> None:
        """Test comparing diffusion coefficients between bulk and grain boundary."""
        shear_dump = _ensure_shear_dump()
        ts = GBStructureTimeseries("ovito", shear_dump)
        ts.perform_cna(enabled=("fcc",), compute=False)

        dt_ps = 2.0  # picoseconds
        with pytest.warns(UserWarning, match="overwrites the timestep property"):
            d_bulk = ts.get_diffusion_coefficient(dt=dt_ps, region="bulk")
        with pytest.warns(UserWarning, match="overwrites the timestep property"):
            d_gb = ts.get_diffusion_coefficient(dt=dt_ps, region="gb")

        assert d_bulk > 0.0
        assert d_gb > 0.0
        # Both D values are in Angstrom^2 / ps
        assert isinstance(d_bulk, float)
        assert isinstance(d_gb, float)
