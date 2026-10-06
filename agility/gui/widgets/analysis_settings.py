# Copyright (c) 2021 Alexander Bonkowski
# Distributed under the terms of the MIT License
# author: Alexander Bonkowski

"""Analysis settings and parameter configuration widget for Agility GUI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget as QWidgetType


class AnalysisSettingsWidget(QWidget):
    """Widget providing method selection and method-specific parameter settings."""

    def __init__(self, parent: QWidgetType | None = None) -> None:
        """Initialize the analysis settings widget."""
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Method selection dropdown
        form = QFormLayout()
        self.method_combo = QComboBox(self)
        self.method_combo.addItems(
            [
                "Common Neighbor Analysis (CNA)",
                "Polyhedral Template Matching (PTM)",
                "Centrosymmetry Parameter (CSP)",
                "Common Neighborhood Parameter (CNP)",
                "Voronoi Analysis",
                "Grain Boundary Fraction",
            ],
        )
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        form.addRow("Method:", self.method_combo)
        layout.addLayout(form)

        # Stacked widget for method-specific parameters
        self.stacked_pages = QStackedWidget(self)
        layout.addWidget(self.stacked_pages)

        # Page 0: CNA
        self._init_cna_page()
        # Page 1: PTM
        self._init_ptm_page()
        # Page 2: CSP
        self._init_csp_page()
        # Page 3: CNP
        self._init_cnp_page()
        # Page 4: Voronoi
        self._init_voronoi_page()
        # Page 5: GB Fraction
        self._init_gb_fraction_page()

    def _init_cna_page(self) -> None:
        """Initialize CNA settings page."""
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()
        self.cna_mode_combo = QComboBox(page)
        self.cna_mode_combo.addItems(
            [
                "IntervalCutoff",
                "AdaptiveCutoff",
                "FixedCutoff",
                "BondBased",
            ],
        )
        form.addRow("CNA Mode:", self.cna_mode_combo)

        self.cna_cutoff_spin = QDoubleSpinBox(page)
        self.cna_cutoff_spin.setRange(0.5, 30.0)
        self.cna_cutoff_spin.setSingleStep(0.1)
        self.cna_cutoff_spin.setValue(3.2)
        form.addRow("Cutoff (Å):", self.cna_cutoff_spin)

        self.cna_color_cb = QCheckBox("Color by type", page)
        self.cna_color_cb.setChecked(True)
        form.addRow(self.cna_color_cb)

        layout.addLayout(form)

        # Structure types group
        group = QGroupBox("Enabled Structures", page)
        grid = QGridLayout(group)
        self.cna_types: dict[str, QCheckBox] = {
            "fcc": QCheckBox("FCC", group),
            "hcp": QCheckBox("HCP", group),
            "bcc": QCheckBox("BCC", group),
            "ico": QCheckBox("ICO", group),
        }
        self.cna_types["fcc"].setChecked(True)
        self.cna_types["hcp"].setChecked(True)
        self.cna_types["bcc"].setChecked(True)
        self.cna_types["ico"].setChecked(False)

        grid.addWidget(self.cna_types["fcc"], 0, 0)
        grid.addWidget(self.cna_types["hcp"], 0, 1)
        grid.addWidget(self.cna_types["bcc"], 1, 0)
        grid.addWidget(self.cna_types["ico"], 1, 1)
        layout.addWidget(group)

        self.stacked_pages.addWidget(page)

    def _init_ptm_page(self) -> None:
        """Initialize PTM settings page."""
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()
        self.ptm_rmsd_spin = QDoubleSpinBox(page)
        self.ptm_rmsd_spin.setRange(0.01, 1.0)
        self.ptm_rmsd_spin.setSingleStep(0.01)
        self.ptm_rmsd_spin.setValue(0.10)
        form.addRow("RMSD Threshold:", self.ptm_rmsd_spin)
        layout.addLayout(form)

        # Structure types group
        group = QGroupBox("Enabled Structures", page)
        grid = QGridLayout(group)
        self.ptm_types: dict[str, QCheckBox] = {
            "fcc": QCheckBox("FCC", group),
            "hcp": QCheckBox("HCP", group),
            "bcc": QCheckBox("BCC", group),
            "ico": QCheckBox("ICO", group),
            "sc": QCheckBox("SC", group),
            "dcub": QCheckBox("Cubic Diamond", group),
            "dhex": QCheckBox("Hex Diamond", group),
            "graphene": QCheckBox("Graphene", group),
        }
        self.ptm_types["fcc"].setChecked(True)
        self.ptm_types["hcp"].setChecked(True)
        self.ptm_types["bcc"].setChecked(True)

        keys = list(self.ptm_types.keys())
        for idx, key in enumerate(keys):
            grid.addWidget(self.ptm_types[key], idx // 2, idx % 2)

        layout.addWidget(group)
        self.stacked_pages.addWidget(page)

    def _init_csp_page(self) -> None:
        """Initialize CSP settings page."""
        page = QWidget(self)
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)

        self.csp_neighbors_spin = QSpinBox(page)
        self.csp_neighbors_spin.setRange(2, 100)
        self.csp_neighbors_spin.setValue(12)
        form.addRow("Nearest Neighbors:", self.csp_neighbors_spin)

        self.stacked_pages.addWidget(page)

    def _init_cnp_page(self) -> None:
        """Initialize CNP settings page."""
        page = QWidget(self)
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)

        self.cnp_cutoff_spin = QDoubleSpinBox(page)
        self.cnp_cutoff_spin.setRange(0.5, 30.0)
        self.cnp_cutoff_spin.setSingleStep(0.1)
        self.cnp_cutoff_spin.setValue(3.2)
        form.addRow("Cutoff (Å):", self.cnp_cutoff_spin)

        self.stacked_pages.addWidget(page)

    def _init_voronoi_page(self) -> None:
        """Initialize Voronoi settings page."""
        page = QWidget(self)
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)

        self.voro_edge_spin = QDoubleSpinBox(page)
        self.voro_edge_spin.setRange(0.0, 10.0)
        self.voro_edge_spin.setSingleStep(0.1)
        self.voro_edge_spin.setValue(0.0)
        form.addRow("Edge Threshold:", self.voro_edge_spin)

        self.voro_radii_cb = QCheckBox("Use atomic radii", page)
        self.voro_radii_cb.setChecked(False)
        form.addRow(self.voro_radii_cb)

        self.stacked_pages.addWidget(page)

    def _init_gb_fraction_page(self) -> None:
        """Initialize Grain Boundary Fraction settings page."""
        page = QWidget(self)
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)

        self.gb_mode_combo = QComboBox(page)
        self.gb_mode_combo.addItems(["cna", "ptm"])
        form.addRow("Filter Mode:", self.gb_mode_combo)

        self.stacked_pages.addWidget(page)

    def _on_method_changed(self, index: int) -> None:
        """Switch parameter page when method selection changes."""
        self.stacked_pages.setCurrentIndex(index)

    def set_method(self, method_key: str) -> None:
        """Select a method by key identifier.

        Args:
            method_key: One of 'cna', 'ptm', 'csp', 'cnp', 'voronoi', 'gb_fraction'.
        """
        mapping = {
            "cna": 0,
            "ptm": 1,
            "csp": 2,
            "cnp": 3,
            "voronoi": 4,
            "gb_fraction": 5,
        }
        idx = mapping.get(method_key, 0)
        self.method_combo.setCurrentIndex(idx)

    def get_current_analysis_params(self) -> tuple[str, dict[str, Any]]:
        """Extract the selected analysis method and its current parameter settings.

        Returns:
            Tuple of (analysis_method_key, kwargs_dictionary).
        """
        idx = self.method_combo.currentIndex()
        if idx == 0:  # CNA
            enabled = [k for k, cb in self.cna_types.items() if cb.isChecked()]
            if not enabled:
                enabled = ["fcc", "hcp", "bcc"]
            return "cna", {
                "mode": self.cna_mode_combo.currentText(),
                "cutoff": self.cna_cutoff_spin.value(),
                "enabled": enabled,
                "color_by_type": self.cna_color_cb.isChecked(),
            }
        if idx == 1:  # PTM
            enabled = [k for k, cb in self.ptm_types.items() if cb.isChecked()]
            if not enabled:
                enabled = ["fcc", "hcp", "bcc"]
            return "ptm", {
                "rmsd_threshold": self.ptm_rmsd_spin.value(),
                "enabled": enabled,
            }
        if idx == 2:  # CSP
            return "csp", {
                "num_neighbors": self.csp_neighbors_spin.value(),
            }
        if idx == 3:  # CNP
            return "cnp", {
                "cutoff": self.cnp_cutoff_spin.value(),
            }
        if idx == 4:  # Voronoi
            return "voronoi", {
                "edge_threshold": self.voro_edge_spin.value(),
                "use_radii": self.voro_radii_cb.isChecked(),
            }
        # GB Fraction
        return "gb_fraction", {
            "mode": self.gb_mode_combo.currentText(),
        }
