"""Integration tests for plotting functions — requires ovito to be installed."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from importlib.util import find_spec
from pathlib import Path
from unittest import SkipTest, TestCase
from urllib.parse import urlparse

import numpy as np
import pytest

from agility.analysis import GBStructure
from agility.plotting import plot_mdf, render_ovito

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILES_DIR = MODULE_DIR.parent / "files"
FIGSHARE_AL_POLYCRYSTAL_URL = "https://figshare.com/ndownloader/files/43058716"


@pytest.mark.integration
@pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
class TestPlotting(TestCase):
    """Test ovito rendering via the plotting module."""

    def setUp(self) -> None:
        """Set up the test."""
        self.data = GBStructure("ovito", f"{TEST_FILES_DIR}/aluminium.lmp")

        assert self.data is not None

    def test_render_ovito(self) -> None:
        """Test the render_ovito function returns a QImage with expected dimensions."""
        image = render_ovito(self.data.pipeline)
        from PySide6.QtGui import QImage  # noqa: PLC0415

        assert isinstance(image, QImage)
        assert image.width() == 282
        assert image.height() == 262

    def test_plot_mdf_cubic_symmetry(self) -> None:
        """Test MDF plotting with cubic symmetry reduction on ovito grain orientations."""
        import matplotlib.figure  # noqa: PLC0415

        self.data.perform_ptm(enabled=("fcc",), output_orientation=True)
        orientations = self.data.get_distinct_grains()
        assert orientations is not None

        fig = plot_mdf(orientations, symmetry="cubic")
        assert isinstance(fig, matplotlib.figure.Figure)
        assert "cubic disorientation" in fig.axes[0].get_title().lower()


@pytest.mark.integration
@pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
class TestMdfLargePolycrystal(TestCase):
    """Integration tests for MDF symmetry behavior on a large Al polycrystal."""

    @staticmethod
    def _make_polycrystal_with_atomsk(output_path: Path) -> bool:
        """Try to create a large (>=100 Å, >=10 grains) Al polycrystal with atomsk."""
        atomsk = shutil.which("atomsk")
        if atomsk is None:
            return False
        atomsk_path = Path(atomsk).resolve()
        if atomsk_path.name != "atomsk" or not atomsk_path.is_file():
            return False

        work_dir = output_path.parent
        unit_cell = work_dir / "al_unitcell.xsf"
        grains = work_dir / "al_grains.txt"
        poly_lmp = work_dir / "al_large_poly.lmp"

        try:
            subprocess.run(  # noqa: S603
                [atomsk, "--create", "fcc", "4.05", "Al", str(unit_cell)],
                check=True,
                cwd=work_dir,
                capture_output=True,
                text=False,
            )
            grains.write_text("box 100 100 100\nrandom 10\n", encoding="utf-8")
            subprocess.run(  # noqa: S603
                [atomsk, "--polycrystal", str(unit_cell), str(grains), str(poly_lmp)],
                check=True,
                cwd=work_dir,
                capture_output=True,
                text=False,
            )
            if poly_lmp.exists():
                output_path.write_bytes(poly_lmp.read_bytes())
                return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

        return False

    @staticmethod
    def _download_polycrystal(output_path: Path) -> None:
        """Download the large Al polycrystal PDB fallback file."""
        parsed = urlparse(FIGSHARE_AL_POLYCRYSTAL_URL)
        if parsed.scheme != "https":
            msg = f"unsupported download URL scheme: {parsed.scheme}"
            raise ValueError(msg)
        with urllib.request.urlopen(FIGSHARE_AL_POLYCRYSTAL_URL, timeout=120) as response:  # noqa: S310
            content_type = response.headers.get("Content-Type", "")
            content_length = response.headers.get("Content-Length")
            if content_type and "text/html" in content_type.lower():
                msg = f"unexpected content type while downloading structure: {content_type}"
                raise ValueError(msg)
            if content_length is not None and int(content_length) > 200_000_000:
                msg = f"downloaded structure is unexpectedly large: {content_length} bytes"
                raise ValueError(msg)
            data = response.read()
            if not data:
                msg = "downloaded structure file is empty"
                raise ValueError(msg)
            output_path.write_bytes(data)

    @classmethod
    def setUpClass(cls) -> None:
        """Create or download a large polycrystalline Al structure for tests."""
        cls._work_dir = Path(tempfile.gettempdir()) / "agility_integration_assets"
        cls._work_dir.mkdir(parents=True, exist_ok=True)
        cls._poly_path = cls._work_dir / "large_al_polycrystal.pdb"

        if not cls._poly_path.exists():
            created = cls._make_polycrystal_with_atomsk(cls._poly_path)
            if not created:
                try:
                    cls._download_polycrystal(cls._poly_path)
                except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                    msg = f"could not create or download large polycrystal: {exc}"
                    raise SkipTest(msg) from exc

    def setUp(self) -> None:
        """Load the large polycrystal structure for analysis."""
        self.data = GBStructure("ovito", str(self._poly_path))
        assert self.data is not None

    @staticmethod
    def _max_nonzero_hist_angle(fig: object) -> float:
        """Return the right edge of the highest-angle non-empty histogram bar."""
        ax = fig.axes[0]
        nonzero = [bar for bar in ax.patches if bar.get_height() > 0.0]
        assert nonzero, "expected at least one non-empty histogram bar"
        return max(bar.get_x() + bar.get_width() for bar in nonzero)

    def test_plot_mdf_symmetry_threshold_on_large_polycrystal(self) -> None:
        """Symmetry reduction should cap cubic disorientation below the threshold."""
        positions = np.asarray(self.data.pipeline.compute().particles.positions)
        span = positions.max(axis=0) - positions.min(axis=0)
        assert np.all(span >= 100.0)

        self.data.perform_ptm(enabled=("fcc",), output_orientation=True)
        orientations = self.data.get_distinct_grains()
        assert orientations is not None
        assert orientations.shape[0] >= 10

        raw_fig = plot_mdf(orientations, bins=360, density=False)
        sym_fig = plot_mdf(orientations, bins=360, density=False, symmetry="cubic")

        threshold_deg = 63.5
        raw_max = self._max_nonzero_hist_angle(raw_fig)
        sym_max = self._max_nonzero_hist_angle(sym_fig)

        assert raw_max > threshold_deg
        assert sym_max <= threshold_deg
