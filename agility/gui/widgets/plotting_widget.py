# Copyright (c) 2021 Alexander Bonkowski
# Distributed under the terms of the MIT License
# author: Alexander Bonkowski

"""Interactive plotting widget for Agility data analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from agility.symmetry import cubic_disorientation_angles

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class PlottingWidget(QWidget):
    """Interactive plotting widget embedding Matplotlib with Qt."""

    plot_requested = Signal(str, dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize plotting widget."""
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Control Bar
        self.control_bar = QWidget(self)
        bar_layout = QHBoxLayout(self.control_bar)
        bar_layout.setContentsMargins(6, 4, 6, 4)

        bar_layout.addWidget(QLabel("Plot Type:", self.control_bar))
        self.plot_type_combo = QComboBox(self.control_bar)
        self.plot_type_combo.addItems(
            [
                "Misorientation Distribution Function (MDF)",
                "Face Order Histogram",
                "Property Distribution",
            ],
        )
        self.plot_type_combo.currentTextChanged.connect(self._on_plot_type_changed)
        bar_layout.addWidget(self.plot_type_combo)

        # Parameter: Bins
        self.bins_label = QLabel("Bins:", self.control_bar)
        bar_layout.addWidget(self.bins_label)
        self.bins_spin = QSpinBox(self.control_bar)
        self.bins_spin.setRange(5, 500)
        self.bins_spin.setValue(30)
        bar_layout.addWidget(self.bins_spin)

        # Parameter: Symmetry
        self.symmetry_label = QLabel("Symmetry:", self.control_bar)
        bar_layout.addWidget(self.symmetry_label)
        self.symmetry_combo = QComboBox(self.control_bar)
        self.symmetry_combo.addItems(["None", "cubic", "hexagonal"])
        bar_layout.addWidget(self.symmetry_combo)

        # Parameter: Density
        self.density_cb = QCheckBox("Density", self.control_bar)
        self.density_cb.setChecked(True)
        bar_layout.addWidget(self.density_cb)

        bar_layout.addStretch()

        self.refresh_btn = QPushButton("Plot", self.control_bar)
        self.refresh_btn.clicked.connect(self._on_refresh_clicked)
        bar_layout.addWidget(self.refresh_btn)

        layout.addWidget(self.control_bar)

        # Matplotlib Canvas & Toolbar
        self.figure: Figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self._current_orientations: np.ndarray | None = None
        self._current_data: Any = None

        self._show_placeholder()

    def _show_placeholder(self) -> None:
        """Display placeholder text when no data is available."""
        self.figure.clear()
        ax: Axes = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "No data plotted yet.\nPerform an analysis or load grain orientations to plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="gray",
            fontsize=11,
        )
        ax.set_axis_off()
        self.canvas.draw_idle()

    def _on_plot_type_changed(self, text: str) -> None:
        """Update visible controls based on selected plot type."""
        is_mdf = "MDF" in text
        self.symmetry_label.setVisible(is_mdf)
        self.symmetry_combo.setVisible(is_mdf)
        self.density_cb.setVisible(is_mdf)

    def _on_refresh_clicked(self) -> None:
        """Emit plot request signal with current control parameters."""
        plot_type = self.plot_type_combo.currentText()
        sym = self.symmetry_combo.currentText()
        params = {
            "bins": self.bins_spin.value(),
            "symmetry": None if sym == "None" else sym,
            "density": self.density_cb.isChecked(),
        }
        self.plot_requested.emit(plot_type, params)

        if "MDF" in plot_type and self._current_orientations is not None:
            self.plot_mdf_data(
                self._current_orientations,
                bins=params["bins"],
                density=params["density"],
                symmetry=params["symmetry"],
            )

    def set_orientations(self, orientations: np.ndarray) -> None:
        """Set orientations data and plot MDF automatically.

        Args:
            orientations: Array of quaternion orientations.
        """
        self._current_orientations = orientations
        sym = self.symmetry_combo.currentText()
        self.plot_mdf_data(
            orientations,
            bins=self.bins_spin.value(),
            density=self.density_cb.isChecked(),
            symmetry=None if sym == "None" else sym,
        )

    def plot_mdf_data(
        self,
        orientations: np.ndarray,
        bins: int = 30,
        density: bool = True,
        symmetry: str | None = None,
    ) -> None:
        """Plot the Misorientation Distribution Function.

        Args:
            orientations: (N, 4) quaternion orientations.
            bins: Number of histogram bins.
            density: Whether to normalize as probability density.
            symmetry: Crystal symmetry ("cubic", "hexagonal", or None).
        """
        q = np.asarray(orientations, dtype=float)
        if q.ndim != 2 or q.shape[1] != 4:
            msg = f"orientations must have shape (N, 4), got {q.shape}"
            raise ValueError(msg)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / norms

        idx_i, idx_j = np.triu_indices(len(q), k=1)
        if len(idx_i) == 0:
            msg = "at least 2 orientations are required to compute pairwise misorientations"
            raise ValueError(msg)

        if symmetry == "cubic":
            angles_deg = cubic_disorientation_angles(q[idx_i], q[idx_j])
            title = "Misorientation Distribution Function (cubic disorientation)"
            xlabel = "Disorientation Angle (°)"
        else:
            dots = np.clip(np.abs(np.einsum("ij,ij->i", q[idx_i], q[idx_j])), 0.0, 1.0)
            angles_deg = np.degrees(2.0 * np.arccos(dots))
            title = "Misorientation Distribution Function (raw, no symmetry reduction)"
            xlabel = "Misorientation Angle (°)"

        self.figure.clear()
        ax: Axes = self.figure.add_subplot(111)
        ax.hist(angles_deg, bins=bins, density=density, edgecolor="black", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability Density" if density else "Count")
        ax.grid(visible=True, linestyle="--", alpha=0.5)
        self.canvas.draw_idle()

    def plot_custom_histogram(
        self,
        values: np.ndarray | list[float],
        title: str = "Distribution",
        xlabel: str = "Value",
        bins: int = 30,
    ) -> None:
        """Plot a general histogram of values.

        Args:
            values: Sequence of numeric values.
            title: Plot title.
            xlabel: X-axis label.
            bins: Number of bins.
        """
        self.figure.clear()
        ax: Axes = self.figure.add_subplot(111)
        ax.hist(values, bins=bins, color="tab:blue", edgecolor="black", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(visible=True, linestyle="--", alpha=0.5)
        self.canvas.draw_idle()

    def clear(self) -> None:
        """Clear plotting canvas."""
        self._current_orientations = None
        self._current_data = None
        self._show_placeholder()
