# Copyright (c) 2021 Alexander Bonkowski
# Distributed under the terms of the MIT License
# author: Alexander Bonkowski

"""Settings dialog for the Agility GUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget as QWidgetType


class SettingsDialog(QDialog):
    """Settings and preferences dialog using QSettings for persistence."""

    def __init__(self, parent: QWidgetType | None = None) -> None:
        """Initialize the settings dialog."""
        super().__init__(parent)
        self.setWindowTitle("Preferences - Agility")
        self.resize(480, 360)
        self.settings = QSettings("Agility", "AgilityGUI")

        layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget(self)
        layout.addWidget(self.tab_widget)

        # Tab 1: General
        self._init_general_tab()
        # Tab 2: Rendering (Molara & 3D)
        self._init_rendering_tab()
        # Tab 3: Plotting
        self._init_plotting_tab()
        # Tab 4: Analysis Defaults
        self._init_analysis_tab()

        # Dialog Buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults,
            self,
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        restore_btn = self.button_box.button(QDialogButtonBox.StandardButton.RestoreDefaults)
        if restore_btn is not None:
            restore_btn.clicked.connect(self.restore_defaults)

        layout.addWidget(self.button_box)

        self.load_settings()

    def _init_general_tab(self) -> None:
        """Initialize general settings tab."""
        widget = QWidget(self)
        layout = QFormLayout(widget)

        self.backend_combo = QComboBox(widget)
        self.backend_combo.addItems(["ovito", "ase", "pymatgen", "lammps"])
        layout.addRow("Default Backend:", self.backend_combo)

        self.auto_render_cb = QCheckBox("Automatically render 3D view on file load", widget)
        layout.addRow(self.auto_render_cb)

        self.tab_widget.addTab(widget, "General")

    def _init_rendering_tab(self) -> None:
        """Initialize 3D rendering settings tab."""
        widget = QWidget(self)
        layout = QFormLayout(widget)

        self.renderer_combo = QComboBox(widget)
        self.renderer_combo.addItems(["molara (if installed)", "matplotlib fallback"])
        layout.addRow("Preferred 3D Engine:", self.renderer_combo)

        self.atom_scale_spin = QDoubleSpinBox(widget)
        self.atom_scale_spin.setRange(0.1, 5.0)
        self.atom_scale_spin.setSingleStep(0.1)
        self.atom_scale_spin.setValue(1.0)
        layout.addRow("Atom Scale Factor:", self.atom_scale_spin)

        self.show_box_cb = QCheckBox("Show simulation box cell boundaries", widget)
        layout.addRow(self.show_box_cb)

        self.tab_widget.addTab(widget, "Rendering")

    def _init_plotting_tab(self) -> None:
        """Initialize data plotting settings tab."""
        widget = QWidget(self)
        layout = QFormLayout(widget)

        self.mdf_bins_spin = QSpinBox(widget)
        self.mdf_bins_spin.setRange(5, 200)
        self.mdf_bins_spin.setValue(30)
        layout.addRow("Default MDF Bins:", self.mdf_bins_spin)

        self.mdf_density_cb = QCheckBox("Plot MDF as probability density", widget)
        self.mdf_density_cb.setChecked(True)
        layout.addRow(self.mdf_density_cb)

        self.tab_widget.addTab(widget, "Plotting")

    def _init_analysis_tab(self) -> None:
        """Initialize analysis default values tab."""
        widget = QWidget(self)
        layout = QFormLayout(widget)

        self.cna_cutoff_spin = QDoubleSpinBox(widget)
        self.cna_cutoff_spin.setRange(0.5, 20.0)
        self.cna_cutoff_spin.setSingleStep(0.1)
        self.cna_cutoff_spin.setValue(3.2)
        layout.addRow("CNA Cutoff Radius (Å):", self.cna_cutoff_spin)

        self.csp_neighbors_spin = QSpinBox(widget)
        self.csp_neighbors_spin.setRange(4, 30)
        self.csp_neighbors_spin.setValue(12)
        layout.addRow("CSP Nearest Neighbors:", self.csp_neighbors_spin)

        self.ptm_rmsd_spin = QDoubleSpinBox(widget)
        self.ptm_rmsd_spin.setRange(0.01, 1.0)
        self.ptm_rmsd_spin.setSingleStep(0.01)
        self.ptm_rmsd_spin.setValue(0.1)
        layout.addRow("PTM RMSD Threshold:", self.ptm_rmsd_spin)

        self.tab_widget.addTab(widget, "Analysis")

    def _get_bool(self, key: str, default: bool) -> bool:
        """Retrieve boolean setting."""
        val = self.settings.value(key, defaultValue=default)
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("true", "1")

    def _get_float(self, key: str, default: float) -> float:
        """Retrieve float setting."""
        val = self.settings.value(key, defaultValue=default)
        return float(str(val))

    def _get_int(self, key: str, default: int) -> int:
        """Retrieve integer setting."""
        val = self.settings.value(key, defaultValue=default)
        return int(str(val))

    def load_settings(self) -> None:
        """Load settings from QSettings into UI controls."""
        backend = str(self.settings.value("general/backend", defaultValue="ovito"))
        index = self.backend_combo.findText(backend)
        if index >= 0:
            self.backend_combo.setCurrentIndex(index)

        self.auto_render_cb.setChecked(self._get_bool("general/auto_render", default=True))

        renderer = str(
            self.settings.value("rendering/engine", defaultValue="molara (if installed)"),
        )
        r_idx = self.renderer_combo.findText(renderer)
        if r_idx >= 0:
            self.renderer_combo.setCurrentIndex(r_idx)

        self.atom_scale_spin.setValue(self._get_float("rendering/atom_scale", default=1.0))
        self.show_box_cb.setChecked(self._get_bool("rendering/show_box", default=True))

        self.mdf_bins_spin.setValue(self._get_int("plotting/mdf_bins", default=30))
        self.mdf_density_cb.setChecked(self._get_bool("plotting/mdf_density", default=True))

        self.cna_cutoff_spin.setValue(self._get_float("analysis/cna_cutoff", default=3.2))
        self.csp_neighbors_spin.setValue(self._get_int("analysis/csp_neighbors", default=12))
        self.ptm_rmsd_spin.setValue(self._get_float("analysis/ptm_rmsd", default=0.1))

    def save_settings(self) -> None:
        """Save settings from UI controls into QSettings."""
        self.settings.setValue("general/backend", self.backend_combo.currentText())
        self.settings.setValue("general/auto_render", self.auto_render_cb.isChecked())
        self.settings.setValue("rendering/engine", self.renderer_combo.currentText())
        self.settings.setValue("rendering/atom_scale", self.atom_scale_spin.value())
        self.settings.setValue("rendering/show_box", self.show_box_cb.isChecked())
        self.settings.setValue("plotting/mdf_bins", self.mdf_bins_spin.value())
        self.settings.setValue("plotting/mdf_density", self.mdf_density_cb.isChecked())
        self.settings.setValue("analysis/cna_cutoff", self.cna_cutoff_spin.value())
        self.settings.setValue("analysis/csp_neighbors", self.csp_neighbors_spin.value())
        self.settings.setValue("analysis/ptm_rmsd", self.ptm_rmsd_spin.value())

    def restore_defaults(self) -> None:
        """Reset controls to default values."""
        self.backend_combo.setCurrentIndex(0)
        self.auto_render_cb.setChecked(True)
        self.renderer_combo.setCurrentIndex(0)
        self.atom_scale_spin.setValue(1.0)
        self.show_box_cb.setChecked(True)
        self.mdf_bins_spin.setValue(30)
        self.mdf_density_cb.setChecked(True)
        self.cna_cutoff_spin.setValue(3.2)
        self.csp_neighbors_spin.setValue(12)
        self.ptm_rmsd_spin.setValue(0.1)

    def accept(self) -> None:
        """Handle OK button click: save settings and close."""
        self.save_settings()
        super().accept()
