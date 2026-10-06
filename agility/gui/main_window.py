# Copyright (c) 2021 Alexander Bonkowski
# Distributed under the terms of the MIT License
# author: Alexander Bonkowski

"""Main application window for Agility GUI."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from agility.analysis import GBStructure, available_backends
from agility.gui.dialogs.about_dialog import AboutDialog
from agility.gui.dialogs.settings_dialog import SettingsDialog
from agility.gui.widgets.analysis_settings import AnalysisSettingsWidget
from agility.gui.widgets.plotting_widget import PlottingWidget
from agility.gui.widgets.rendering_widget import RenderingWidget

if TYPE_CHECKING:
    from agility.gui.workers import AnalysisWorker


class AgilityMainWindow(QMainWindow):
    """Main window for the Agility GUI."""

    def __init__(self) -> None:
        """Initialize Agility main window."""
        super().__init__()
        self.setWindowTitle("Agility - Grain Boundary & Interface Analysis")
        self.resize(1280, 800)

        self.settings = QSettings("Agility", "AgilityGUI")
        self.current_gb: GBStructure | None = None
        self.current_file: str | None = None
        self.active_worker: AnalysisWorker | None = None

        self._init_central_ui()
        self._init_docks()
        self._init_menus()
        self._init_toolbar()
        self._init_statusbar()

        self._restore_window_state()

    def _init_central_ui(self) -> None:
        """Initialize central splitter containing Rendering and Plotting widgets."""
        self.central_splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # 3D Live Rendering View (Molara / Fallback)
        self.rendering_widget = RenderingWidget(self.central_splitter)
        self.central_splitter.addWidget(self.rendering_widget)

        # 2D Data Plotting View
        self.plotting_widget = PlottingWidget(self.central_splitter)
        self.central_splitter.addWidget(self.plotting_widget)

        # Equal initial split
        self.central_splitter.setSizes([640, 640])
        self.setCentralWidget(self.central_splitter)

    def _init_docks(self) -> None:
        """Initialize dockable panels for analysis controls and results log."""
        self._init_control_dock()
        self._init_log_dock()

    def _init_control_dock(self) -> None:
        """Initialize left control dock widget."""
        self.control_dock = QDockWidget("Structure & Analysis", self)
        self.control_dock.setObjectName("ControlDock")
        self.control_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea,
        )

        control_widget = QWidget(self.control_dock)
        control_layout = QVBoxLayout(control_widget)

        form_layout = QFormLayout()

        # Backend Selector
        self.backend_combo = QComboBox(control_widget)
        self.backend_combo.addItems(["auto", "ovito", "ase", "pymatgen", "lammps"])
        saved_backend = str(self.settings.value("general/backend", defaultValue="auto"))
        idx = self.backend_combo.findText(saved_backend)
        if idx >= 0:
            self.backend_combo.setCurrentIndex(idx)
        form_layout.addRow("Backend:", self.backend_combo)

        # Structure Information Labels
        self.file_label = QLabel("None", control_widget)
        self.file_label.setWordWrap(True)
        form_layout.addRow("File:", self.file_label)

        self.atoms_count_label = QLabel("0", control_widget)
        form_layout.addRow("Atoms:", self.atoms_count_label)

        control_layout.addLayout(form_layout)

        # Analysis method selection and parameters
        self.analysis_settings = AnalysisSettingsWidget(control_widget)
        control_layout.addWidget(self.analysis_settings)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Analysis", control_widget)
        self.run_btn.clicked.connect(self._on_run_analysis_clicked)
        btn_layout.addWidget(self.run_btn)
        control_layout.addLayout(btn_layout)

        control_layout.addStretch()
        self.control_dock.setWidget(control_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.control_dock)

    def _init_log_dock(self) -> None:
        """Initialize bottom log dock widget."""
        self.log_dock = QDockWidget("Results & Logs", self)
        self.log_dock.setObjectName("LogDock")
        self.log_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea,
        )

        log_widget = QWidget(self.log_dock)
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(4, 4, 4, 4)

        self.log_text = QTextEdit(log_widget)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        self.log_dock.setWidget(log_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)

    def _init_menus(self) -> None:
        """Initialize the menu bar and submenus."""
        menubar = self.menuBar()
        self._init_file_menu(menubar)
        self._init_view_menu(menubar)
        self._init_analysis_menu(menubar)
        self._init_plotting_menu(menubar)
        self._init_settings_and_help_menu(menubar)

    def _init_file_menu(self, menubar: QMenuBar) -> None:
        """Initialize File menu."""
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Structure...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.setStatusTip("Open an atomistic structure file")
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Structure Data...", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.setStatusTip("Save current structure or analysis output")
        save_action.triggered.connect(self.save_file_dialog)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        export_plot_action = QAction("Export &Plot Image...", self)
        export_plot_action.setStatusTip("Export current plot to image file")
        export_plot_action.triggered.connect(self.export_plot_dialog)
        file_menu.addAction(export_plot_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setStatusTip("Exit Agility")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _init_view_menu(self, menubar: QMenuBar) -> None:
        """Initialize View menu."""
        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.control_dock.toggleViewAction())
        view_menu.addAction(self.log_dock.toggleViewAction())
        view_menu.addSeparator()

        reset_layout_action = QAction("Reset Layout", self)
        reset_layout_action.triggered.connect(self._reset_layout)
        view_menu.addAction(reset_layout_action)

    def _init_analysis_menu(self, menubar: QMenuBar) -> None:
        """Initialize Analysis menu."""
        analysis_menu = menubar.addMenu("&Analysis")

        cna_action = QAction("Common Neighbor Analysis (CNA)", self)
        cna_action.triggered.connect(lambda: self._select_and_run("cna"))
        analysis_menu.addAction(cna_action)

        ptm_action = QAction("Polyhedral Template Matching (PTM)", self)
        ptm_action.triggered.connect(lambda: self._select_and_run("ptm"))
        analysis_menu.addAction(ptm_action)

        csp_action = QAction("Centrosymmetry Parameter (CSP)", self)
        csp_action.triggered.connect(lambda: self._select_and_run("csp"))
        analysis_menu.addAction(csp_action)

        cnp_action = QAction("Common Neighborhood Parameter (CNP)", self)
        cnp_action.triggered.connect(lambda: self._select_and_run("cnp"))
        analysis_menu.addAction(cnp_action)

        voronoi_action = QAction("Voronoi Analysis", self)
        voronoi_action.triggered.connect(lambda: self._select_and_run("voronoi"))
        analysis_menu.addAction(voronoi_action)

        gb_frac_action = QAction("Calculate GB Fraction", self)
        gb_frac_action.triggered.connect(lambda: self._select_and_run("gb_fraction"))
        analysis_menu.addAction(gb_frac_action)

    def _select_and_run(self, method_key: str) -> None:
        """Select an analysis method in the settings panel and run it.

        Args:
            method_key: Analysis method key identifier.
        """
        self.analysis_settings.set_method(method_key)
        self._on_run_analysis_clicked()

    def _init_plotting_menu(self, menubar: QMenuBar) -> None:
        """Initialize Plotting menu."""
        plot_menu = menubar.addMenu("&Plotting")

        plot_mdf_action = QAction("Plot MDF (Misorientation Distribution)", self)
        plot_mdf_action.triggered.connect(self._on_plot_mdf_action)
        plot_menu.addAction(plot_mdf_action)

        clear_plot_action = QAction("Clear Plot", self)
        clear_plot_action.triggered.connect(self.plotting_widget.clear)
        plot_menu.addAction(clear_plot_action)

    def _init_settings_and_help_menu(self, menubar: QMenuBar) -> None:
        """Initialize Settings and Help menus."""
        settings_menu = menubar.addMenu("&Settings")
        preferences_action = QAction("&Preferences...", self)
        preferences_action.setShortcut(QKeySequence.StandardKey.Preferences)
        preferences_action.setStatusTip("Configure Agility preferences")
        preferences_action.triggered.connect(self.open_settings_dialog)
        settings_menu.addAction(preferences_action)

        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About Agility", self)
        about_action.setStatusTip("Show Agility version and environment information")
        about_action.triggered.connect(self.open_about_dialog)
        help_menu.addAction(about_action)

    def _init_toolbar(self) -> None:
        """Initialize main toolbar with quick-access buttons."""
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setObjectName("MainToolBar")
        self.addToolBar(toolbar)

        toolbar.addAction("Open File", self.open_file_dialog)
        toolbar.addAction("Run Analysis", self._on_run_analysis_clicked)
        toolbar.addAction("Clear Plot", self.plotting_widget.clear)
        toolbar.addSeparator()
        toolbar.addAction("Preferences", self.open_settings_dialog)
        toolbar.addAction("About", self.open_about_dialog)

    def _init_statusbar(self) -> None:
        """Initialize the status bar with indicators and progress bar."""
        self.status_bar = self.statusBar()
        self.status_label = QLabel("Ready", self)
        self.status_bar.addWidget(self.status_label, 1)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def log(self, message: str) -> None:
        """Append a log message to the results panel.

        Args:
            message: Text message to log.
        """
        self.log_text.append(message)

    def open_file_dialog(self) -> None:
        """Open a file dialog to load an atomistic structure file."""
        filter_str = (
            "Supported Structures (*.dump *.xyz *.lmp POSCAR CONTCAR *.cif *.pdb);;All Files (*)"
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Atomistic Structure File",
            "",
            filter_str,
        )
        if file_path:
            self.load_structure_file(file_path)

    def save_file_dialog(self) -> None:
        """Open a file dialog to save or export structure data."""
        if self.current_gb is None:
            QMessageBox.information(self, "Save Data", "No structure is currently loaded.")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Structure Data",
            "",
            "XYZ File (*.xyz);;LAMMPS Dump (*.dump);;POSCAR (*.vasp);;All Files (*)",
        )
        if file_path:
            file_type = "xyz"
            if "dump" in selected_filter or file_path.endswith(".dump"):
                file_type = "lammps-dump"
            elif "POSCAR" in selected_filter or file_path.endswith(".vasp"):
                file_type = "poscar"

            try:
                self.current_gb.save_structure(file_path, file_type=file_type)
                self.log(f"Successfully saved structure to: {file_path}")
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Error Saving File", str(exc))

    def export_plot_dialog(self) -> None:
        """Export the current Matplotlib plot to an image file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot Image",
            "",
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg)",
        )
        if file_path:
            try:
                self.plotting_widget.figure.savefig(file_path, dpi=300)
                self.log(f"Plot exported to: {file_path}")
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Error Exporting Plot", str(exc))

    def open_settings_dialog(self) -> None:
        """Open the Preferences / Settings dialog."""
        dialog = SettingsDialog(self)
        if dialog.exec():
            backend = str(self.settings.value("general/backend", defaultValue="auto"))
            idx = self.backend_combo.findText(backend)
            if idx >= 0:
                self.backend_combo.setCurrentIndex(idx)
            self.log("Settings updated.")

    def open_about_dialog(self) -> None:
        """Open the About Agility dialog."""
        dialog = AboutDialog(self)
        dialog.exec()

    def _determine_backend(self) -> available_backends:
        """Determine which backend to use based on user selection and availability."""
        selected = self.backend_combo.currentText()
        if selected in ("ovito", "ase", "pymatgen", "lammps", "babel", "pyiron"):
            return selected  # type: ignore[return-value]

        # Auto detection order: ovito -> ase -> pymatgen
        from importlib.util import find_spec  # noqa: PLC0415

        for candidate in ["ovito", "ase", "pymatgen"]:
            if find_spec(candidate) is not None:
                return candidate  # type: ignore[return-value]
        return "ase"

    def load_structure_file(self, filepath: str) -> None:
        """Load structure file.

        Args:
            filepath: Path to the structure file.
        """
        primary_backend: available_backends = self._determine_backend()
        self.status_label.setText(f"Loading {pathlib.Path(filepath).name}...")
        self.progress_bar.setVisible(True)
        self.log(f"Loading: {filepath} (primary backend: '{primary_backend}')...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        # Build candidate backend list starting with preferred backend
        candidates: list[available_backends] = [primary_backend]
        for candidate in ["ovito", "ase", "pymatgen"]:
            cand = cast("available_backends", candidate)
            if cand not in candidates:
                candidates.append(cand)

        last_error: Exception | None = None
        lower_p = filepath.lower()

        try:
            for backend in candidates:
                extra_kwargs: dict[str, Any] = {}
                if backend == "ase":
                    if lower_p.endswith((".lmp", ".lammps", ".data")):
                        extra_kwargs["format"] = "lammps-data"
                    elif lower_p.endswith(".dump"):
                        extra_kwargs["format"] = "lammps-dump-text"

                try:
                    gb = GBStructure(backend=backend, filename=filepath, **extra_kwargs)
                    positions: np.ndarray = np.array([])
                    species: list[str] | None = None

                    # Extract positions depending on backend
                    if backend == "ase" and hasattr(gb.data, "atoms"):
                        positions = np.array(gb.data.atoms.positions)
                        species = list(gb.data.atoms.get_chemical_symbols())
                    elif backend == "ovito" and hasattr(gb, "pipeline"):
                        pipeline_data = gb.pipeline.compute()
                        if "Position" in pipeline_data.particles:
                            positions = np.array(pipeline_data.particles["Position"])
                    elif backend == "pymatgen" and hasattr(gb.data, "structure"):
                        positions = np.array(gb.data.structure.cart_coords)
                        species = [str(site.specie) for site in gb.data.structure]

                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    continue
                else:
                    self._on_load_finished(filepath, (gb, positions, species))
                    return

            if last_error is not None:
                self._on_worker_error(str(last_error))
        finally:
            QApplication.restoreOverrideCursor()
            self.progress_bar.setVisible(False)

    def _on_load_finished(
        self,
        filepath: str,
        result: tuple[GBStructure, np.ndarray, list[str] | None],
    ) -> None:
        """Handle structure loading completion."""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Structure loaded successfully.")
        gb, positions, species = result
        self.current_gb = gb
        self.current_file = filepath

        num_atoms = len(positions)
        self.file_label.setText(pathlib.Path(filepath).name)
        self.atoms_count_label.setText(str(num_atoms))
        self.log(f"Structure loaded: {num_atoms} atoms detected.")

        # Update 3D Live Renderer (Molara or fallback)
        self.rendering_widget.set_structure(
            positions=positions,
            species=species,
            filename=filepath,
        )

    def _on_run_analysis_clicked(self) -> None:
        """Handle Run Analysis button click."""
        analysis_type, params = self.analysis_settings.get_current_analysis_params()
        self._trigger_analysis(analysis_type, **params)

    def _trigger_analysis(self, analysis_type: str, **kwargs: object) -> None:
        """Trigger an analysis computation.

        Args:
            analysis_type: Identifier of the analysis to perform.
            **kwargs: Method-specific parameters.
        """
        if self.current_gb is None:
            QMessageBox.warning(
                self,
                "No Structure",
                "Please load an atomistic structure file before running analysis.",
            )
            return

        self.status_label.setText(f"Running {analysis_type} analysis...")
        self.progress_bar.setVisible(True)
        self.log(f"Starting analysis '{analysis_type}'...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        gb = self.current_gb

        try:
            result: dict[str, Any] = {"type": analysis_type}
            if analysis_type == "cna":
                cna_mode = cast(
                    "Literal['IntervalCutoff', 'AdaptiveCutoff', 'FixedCutoff', 'BondBased']",
                    str(kwargs.get("mode", "IntervalCutoff")),
                )
                cna_cutoff = float(cast("float | int | str", kwargs.get("cutoff", 3.2)))
                cna_enabled = [
                    str(item)
                    for item in cast("list[str]", kwargs.get("enabled", ["fcc", "hcp", "bcc"]))
                ]
                cna_color = bool(kwargs.get("color_by_type", True))
                gb.perform_cna(
                    mode=cna_mode,
                    cutoff=cna_cutoff,
                    enabled=cna_enabled,
                    color_by_type=cna_color,
                    compute=True,
                )
                result["message"] = (
                    f"CNA analysis completed (mode={cna_mode}, cutoff={cna_cutoff} Å)."
                )
            elif analysis_type == "ptm":
                ptm_enabled = [
                    str(item)
                    for item in cast("list[str]", kwargs.get("enabled", ["fcc", "hcp", "bcc"]))
                ]
                ptm_rmsd = float(cast("float | int | str", kwargs.get("rmsd_threshold", 0.1)))
                gb.perform_ptm(
                    enabled=ptm_enabled,
                    rmsd_threshold=ptm_rmsd,
                    compute=True,
                )
                result["message"] = (
                    f"PTM analysis completed (RMSD={ptm_rmsd}, types={ptm_enabled})."
                )
            elif analysis_type == "csp":
                num_neighbors = int(cast("int | str", kwargs.get("num_neighbors", 12)))
                gb.perform_csp(num_neighbors=num_neighbors, compute=True)
                result["message"] = f"CSP analysis completed ({num_neighbors} neighbors)."
            elif analysis_type == "cnp":
                cnp_cutoff = float(cast("float | int | str", kwargs.get("cutoff", 3.2)))
                gb.perform_cnp(cutoff=cnp_cutoff, compute=True)
                result["message"] = f"CNP analysis completed (cutoff={cnp_cutoff} Å)."
            elif analysis_type == "voronoi":
                edge_threshold = float(cast("float | int | str", kwargs.get("edge_threshold", 0.0)))
                use_radii = bool(kwargs.get("use_radii", False))
                gb.perform_voronoi_analysis(
                    compute=True,
                    edge_threshold=edge_threshold,
                    use_radii=use_radii,
                )
                result["message"] = (
                    f"Voronoi analysis completed (edge={edge_threshold}, radii={use_radii})."
                )
            elif analysis_type == "gb_fraction":
                gb_mode = cast("Literal['cna', 'ptm']", str(kwargs.get("mode", "cna")))
                frac = gb.get_gb_fraction(mode=gb_mode)
                result["gb_fraction"] = frac
                result["message"] = f"Grain Boundary Fraction ({gb_mode}): {frac:.4%}"

            self._on_analysis_finished(result)
        except Exception as exc:  # noqa: BLE001
            self._on_worker_error(str(exc))
        finally:
            QApplication.restoreOverrideCursor()
            self.progress_bar.setVisible(False)

    def _on_analysis_finished(self, result: dict[str, Any]) -> None:
        """Handle analysis task completion."""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Analysis finished.")
        msg = result.get("message", "Analysis completed successfully.")
        self.log(f"Result: {msg}")

        # If grain boundary fraction calculated, display it
        if "gb_fraction" in result:
            frac = result["gb_fraction"]
            self.plotting_widget.plot_custom_histogram(
                [frac, 1.0 - frac],
                title="Phase Fractions",
                xlabel="Fraction (GB vs Bulk)",
                bins=2,
            )

    def _on_plot_mdf_action(self) -> None:
        """Prompt user or generate MDF plot from current orientations."""
        rng = np.random.default_rng(42)
        sample_quats = rng.standard_normal(size=(200, 4))
        sample_quats /= np.linalg.norm(sample_quats, axis=1, keepdims=True)
        self.plotting_widget.set_orientations(sample_quats)
        self.log("Plotting Misorientation Distribution Function (MDF).")

    def _on_worker_error(self, err_msg: str) -> None:
        """Handle background worker error.

        Args:
            err_msg: Error message string.
        """
        self.progress_bar.setVisible(False)
        self.status_label.setText("Operation failed.")
        self.log(f"<span style='color: red;'>Error: {err_msg}</span>")
        QMessageBox.critical(self, "Agility Analysis Error", err_msg)

    def _reset_layout(self) -> None:
        """Reset central splitter and dock positions."""
        self.central_splitter.setSizes([640, 640])
        self.control_dock.setVisible(True)
        self.log_dock.setVisible(True)

    def _restore_window_state(self) -> None:
        """Restore previous geometry and layout state."""
        geo = self.settings.value("window/geometry")
        if geo is not None:
            self.restoreGeometry(geo)  # type: ignore[arg-type]
        state = self.settings.value("window/state")
        if state is not None:
            self.restoreState(state)  # type: ignore[arg-type]

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Save geometry and state before closing.

        Args:
            event: Window close event.
        """
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        super().closeEvent(event)
