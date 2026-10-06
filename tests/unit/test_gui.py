"""Unit tests for Agility GUI components."""

from __future__ import annotations

import os
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from agility.gui.app import main, parse_args
from agility.gui.dialogs.about_dialog import AboutDialog
from agility.gui.dialogs.settings_dialog import SettingsDialog
from agility.gui.main_window import AgilityMainWindow
from agility.gui.widgets.analysis_settings import AnalysisSettingsWidget
from agility.gui.widgets.plotting_widget import PlottingWidget
from agility.gui.widgets.rendering_widget import RenderingWidget
from agility.gui.workers import AnalysisWorker

os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="session", autouse=True)
def _ensure_qapp() -> QApplication:
    """Ensure a headless QApplication exists for all tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["agility-gui-test"])
    app._agility_test_mode = True  # noqa: SLF001
    return app


@pytest.mark.unit
class TestSettingsDialog(TestCase):
    """Unit tests for SettingsDialog."""

    def test_settings_dialog_lifecycle(self) -> None:
        """Test dialog creation, loading, saving, and defaults restoration."""
        dialog = SettingsDialog()
        assert dialog.windowTitle() == "Preferences - Agility"

        # Change settings values
        dialog.backend_combo.setCurrentText("ase")
        dialog.auto_render_cb.setChecked(False)
        dialog.atom_scale_spin.setValue(1.5)
        dialog.show_box_cb.setChecked(False)
        dialog.mdf_bins_spin.setValue(50)
        dialog.cna_cutoff_spin.setValue(4.0)
        dialog.ptm_rmsd_spin.setValue(0.15)

        # Save settings
        dialog.save_settings()

        settings = QSettings("Agility", "AgilityGUI")
        assert settings.value("general/backend") == "ase"
        assert settings.value("general/auto_render", defaultValue=True, type=bool) is False
        assert float(str(settings.value("rendering/atom_scale"))) == 1.5
        assert float(str(settings.value("analysis/ptm_rmsd"))) == 0.15

        # Test restore defaults
        dialog.restore_defaults()
        assert dialog.backend_combo.currentText() == "ovito"
        assert dialog.auto_render_cb.isChecked() is True
        assert dialog.mdf_bins_spin.value() == 30
        assert dialog.ptm_rmsd_spin.value() == 0.10

        dialog.save_settings()
        settings.setValue("general/backend", "auto")
        dialog.close()


@pytest.mark.unit
class TestAboutDialog(TestCase):
    """Unit tests for AboutDialog."""

    def test_about_dialog_initialization(self) -> None:
        """Test AboutDialog initialization and package check."""
        dialog = AboutDialog()
        assert "Agility" in dialog.windowTitle()
        status = dialog._check_dependencies()  # noqa: SLF001
        assert "ase" in status
        assert "molara" in status
        dialog.close()


@pytest.mark.unit
class TestAnalysisSettingsWidget(TestCase):
    """Unit tests for AnalysisSettingsWidget."""

    def test_cna_settings(self) -> None:
        """Test CNA parameter configuration."""
        widget = AnalysisSettingsWidget()
        widget.set_method("cna")

        widget.cna_mode_combo.setCurrentText("AdaptiveCutoff")
        widget.cna_cutoff_spin.setValue(2.8)
        widget.cna_types["fcc"].setChecked(True)
        widget.cna_types["hcp"].setChecked(False)
        widget.cna_types["bcc"].setChecked(True)
        widget.cna_types["ico"].setChecked(True)
        widget.cna_color_cb.setChecked(False)

        method, params = widget.get_current_analysis_params()
        assert method == "cna"
        assert params["mode"] == "AdaptiveCutoff"
        assert params["cutoff"] == 2.8
        assert "fcc" in params["enabled"]
        assert "hcp" not in params["enabled"]
        assert "ico" in params["enabled"]
        assert params["color_by_type"] is False

    def test_ptm_settings(self) -> None:
        """Test PTM parameter configuration."""
        widget = AnalysisSettingsWidget()
        widget.set_method("ptm")

        widget.ptm_rmsd_spin.setValue(0.18)
        widget.ptm_types["fcc"].setChecked(True)
        widget.ptm_types["bcc"].setChecked(False)
        widget.ptm_types["graphene"].setChecked(True)

        method, params = widget.get_current_analysis_params()
        assert method == "ptm"
        assert params["rmsd_threshold"] == 0.18
        assert "fcc" in params["enabled"]
        assert "bcc" not in params["enabled"]
        assert "graphene" in params["enabled"]

    def test_csp_settings(self) -> None:
        """Test CSP parameter configuration."""
        widget = AnalysisSettingsWidget()
        widget.set_method("csp")
        widget.csp_neighbors_spin.setValue(16)

        method, params = widget.get_current_analysis_params()
        assert method == "csp"
        assert params["num_neighbors"] == 16

    def test_cnp_settings(self) -> None:
        """Test CNP parameter configuration."""
        widget = AnalysisSettingsWidget()
        widget.set_method("cnp")
        widget.cnp_cutoff_spin.setValue(3.4)

        method, params = widget.get_current_analysis_params()
        assert method == "cnp"
        assert params["cutoff"] == 3.4

    def test_voronoi_settings(self) -> None:
        """Test Voronoi parameter configuration."""
        widget = AnalysisSettingsWidget()
        widget.set_method("voronoi")
        widget.voro_edge_spin.setValue(0.2)
        widget.voro_radii_cb.setChecked(True)

        method, params = widget.get_current_analysis_params()
        assert method == "voronoi"
        assert params["edge_threshold"] == 0.2
        assert params["use_radii"] is True

    def test_gb_fraction_settings(self) -> None:
        """Test GB Fraction parameter configuration."""
        widget = AnalysisSettingsWidget()
        widget.set_method("gb_fraction")
        widget.gb_mode_combo.setCurrentText("ptm")

        method, params = widget.get_current_analysis_params()
        assert method == "gb_fraction"
        assert params["mode"] == "ptm"


@pytest.mark.unit
class TestRenderingWidget(TestCase):
    """Unit tests for RenderingWidget."""

    def test_rendering_widget_structure(self) -> None:
        """Test setting and clearing atomic structures in rendering widget."""
        widget = RenderingWidget()
        assert widget.status_label is not None

        positions = np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]])
        species = ["Fe", "Fe"]
        widget.set_structure(positions=positions, species=species, filename="test.xyz")

        assert widget._current_positions is not None  # noqa: SLF001
        assert len(widget._current_positions) == 2  # noqa: SLF001

        # Test clear
        widget.clear()
        assert widget._current_positions is None  # noqa: SLF001

        widget.close()

    def test_launch_molara_handling(self) -> None:
        """Test molara launch handling when package is mocked."""
        widget = RenderingWidget()
        with (
            patch("agility.gui.widgets.rendering_widget.HAS_MOLARA", new=True),
            patch.dict("sys.modules", {"molara": MagicMock()}),
        ):
            widget._current_filename = "structure.dump"  # noqa: SLF001
            widget._launch_molara()  # noqa: SLF001

        widget.close()


@pytest.mark.unit
class TestPlottingWidget(TestCase):
    """Unit tests for PlottingWidget."""

    def test_plotting_widget_mdf(self) -> None:
        """Test MDF plotting and orientations update."""
        widget = PlottingWidget()

        rng = np.random.default_rng(123)
        quats = rng.standard_normal(size=(50, 4))
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)

        widget.set_orientations(quats)
        assert widget._current_orientations is not None  # noqa: SLF001

        # Test plot type combo switch
        widget.plot_type_combo.setCurrentText("Face Order Histogram")
        assert widget.symmetry_combo.isVisible() is False

        # Test custom histogram
        widget.plot_custom_histogram([0.1, 0.2, 0.3], title="Test Histogram")

        # Test clear
        widget.clear()
        assert widget._current_orientations is None  # noqa: SLF001

        widget.close()


@pytest.mark.unit
class TestAnalysisWorker(TestCase):
    """Unit tests for background AnalysisWorker."""

    def test_worker_success(self) -> None:
        """Test successful execution of worker callable."""

        def add(a: int, b: int) -> int:
            return a + b

        worker = AnalysisWorker(add, 10, 20)
        results: list[int] = []
        worker.signals.finished.connect(results.append)
        worker.run()
        assert results == [30]

    def test_worker_error(self) -> None:
        """Test error propagation in worker."""

        def fail() -> None:
            msg = "Deliberate failure"
            raise ValueError(msg)

        worker = AnalysisWorker(fail)
        errors: list[str] = []
        worker.signals.error.connect(errors.append)
        worker.run()
        assert len(errors) == 1
        assert "Deliberate failure" in errors[0]


@pytest.mark.unit
class TestAgilityMainWindow(TestCase):
    """Unit tests for AgilityMainWindow."""

    def test_main_window_creation_and_layout(self) -> None:
        """Test main window initialization, docks, menus and actions."""
        window = AgilityMainWindow()
        assert "Agility" in window.windowTitle()
        assert window.control_dock is not None
        assert window.log_dock is not None
        assert window.rendering_widget is not None
        assert window.plotting_widget is not None
        assert window.analysis_settings is not None

        # Test logging
        window.log("Test log entry")
        assert "Test log entry" in window.log_text.toPlainText()

        # Test backend detection
        backend = window._determine_backend()  # noqa: SLF001
        assert backend in ("ovito", "ase", "pymatgen", "lammps", "babel", "pyiron")

        # Test reset layout
        window._reset_layout()  # noqa: SLF001

        # Test MDF action
        window._on_plot_mdf_action()  # noqa: SLF001

        window.close()


@pytest.mark.unit
class TestAppLauncher(TestCase):
    """Unit tests for Agility CLI launcher."""

    def test_parse_args(self) -> None:
        """Test CLI argument parsing."""
        args = parse_args(["structure.dump"])
        assert args.structure_file == "structure.dump"

        empty_args = parse_args([])
        assert empty_args.structure_file is None

    def test_main_entrypoint(self) -> None:
        """Test main entrypoint in test mode."""
        exit_code = main([])
        assert exit_code == 0
