# Copyright (c) 2021 Alexander Bonkowski
# Distributed under the terms of the MIT License
# author: Alexander Bonkowski

"""3D Live Rendering widget for Agility, supporting Molara as an optional dependency."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.figure import Figure

HAS_MOLARA = find_spec("molara") is not None


class RenderingWidget(QWidget):
    """3D structure visualization widget.

    Uses `molara` for live rendering if installed. Otherwise provides an informative
    fallback message and a built-in Matplotlib 3D scatter view.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the rendering widget."""
        super().__init__(parent)
        self.molara_instance: Any = None
        self.ax: Any = None
        self._current_positions: np.ndarray | None = None
        self._current_species: list[str] | np.ndarray | None = None
        self._current_cell: np.ndarray | None = None
        self._current_filename: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header bar showing active renderer status
        self.header_widget = QWidget(self)
        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(6, 4, 6, 4)

        if HAS_MOLARA:
            status_text = (
                "<b>3D View:</b> <span style='color: green;'>Molara Engine (Active)</span>"
            )
        else:
            status_text = (
                "<b>3D View:</b> <span style='color: orange;'>"
                "Fallback (molara not installed)</span>"
            )
        self.status_label = QLabel(status_text, self.header_widget)
        header_layout.addWidget(self.status_label)

        header_layout.addStretch()

        self.molara_btn = QPushButton("Launch Molara Live Viewer", self.header_widget)
        self.molara_btn.setEnabled(HAS_MOLARA)
        self.molara_btn.clicked.connect(self._launch_molara)
        header_layout.addWidget(self.molara_btn)

        layout.addWidget(self.header_widget)

        # Stacked view: Molara container vs Fallback view
        self.stack = QStackedWidget(self)
        layout.addWidget(self.stack)

        self._init_fallback_view()
        self._init_molara_view()

        if HAS_MOLARA:
            self.stack.setCurrentIndex(1)
        else:
            self.stack.setCurrentIndex(0)

    def _init_fallback_view(self) -> None:
        """Initialize the fallback Matplotlib 3D visualization view."""
        self.fallback_container = QWidget(self)
        vbox = QVBoxLayout(self.fallback_container)
        vbox.setContentsMargins(8, 8, 8, 8)

        # Optional dependency notice
        self.notice_label = QLabel(
            "<div style='background-color: #f0f4f8; border: 1px solid #cbd5e1; "
            "border-radius: 6px; padding: 10px; margin-bottom: 8px;'>"
            "<b>Optional Dependency Notice:</b> Live advanced 3D rendering uses the "
            "<code>molara</code> package.<br>"
            "To install it, run: <code>pip install molara</code>.<br>"
            "Displaying built-in 3D preview below:"
            "</div>",
            self.fallback_container,
        )
        self.notice_label.setTextFormat(Qt.TextFormat.RichText)
        self.notice_label.setWordWrap(True)
        vbox.addWidget(self.notice_label)

        # Matplotlib 3D Canvas
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: PLC0415
        from matplotlib.figure import Figure  # noqa: PLC0415

        self.fig: Figure = Figure(figsize=(5, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        if self.ax is not None:
            self.ax.set_title("No structure loaded")
            self.ax.set_xlabel("X (Å)")
            self.ax.set_ylabel("Y (Å)")
            self.ax.set_zlabel("Z (Å)")  # type: ignore[attr-defined]

        vbox.addWidget(self.canvas)
        self.stack.addWidget(self.fallback_container)

    def _init_molara_view(self) -> None:
        """Initialize the container for Molara."""
        self.molara_container = QWidget(self)
        vbox = QVBoxLayout(self.molara_container)
        vbox.setContentsMargins(8, 8, 8, 8)

        self.molara_info_label = QLabel(
            "<h3>Molara 3D Live Renderer</h3>"
            "<p>Structure will be rendered with Molara. Click 'Launch Molara Live Viewer' "
            "or load a structure file to inspect interactively.</p>",
            self.molara_container,
        )
        self.molara_info_label.setTextFormat(Qt.TextFormat.RichText)
        self.molara_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(self.molara_info_label)

        self.stack.addWidget(self.molara_container)

    def is_molara_available(self) -> bool:
        """Return True if molara package is installed."""
        return HAS_MOLARA

    def set_structure(
        self,
        positions: np.ndarray,
        species: list[str] | np.ndarray | None = None,
        cell: np.ndarray | None = None,
        filename: str | None = None,
    ) -> None:
        """Update the rendering widget with a new atomic structure.

        Args:
            positions: (N, 3) array of atomic coordinates.
            species: Optional list or array of element symbols.
            cell: Optional 3x3 lattice vectors or bounding box.
            filename: Optional source filename.
        """
        self._current_positions = positions
        self._current_species = species
        self._current_cell = cell
        self._current_filename = filename

        # Update fallback 3D preview
        if self.ax is not None:
            self.ax.clear()
            if len(positions) > 0:
                # Limit scatter points if atom count is huge for responsiveness
                max_pts = 5000
                pts = positions if len(positions) <= max_pts else positions[:max_pts]
                self.ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    s=20,
                    alpha=0.8,
                    c="tab:blue",
                    edgecolors="none",
                )
                self.ax.set_title(
                    f"Structure: {len(positions)} atoms"
                    + (f" (showing first {max_pts})" if len(positions) > max_pts else ""),
                )
            else:
                self.ax.set_title("Empty structure")

            self.ax.set_xlabel("X (Å)")
            self.ax.set_ylabel("Y (Å)")
            self.ax.set_zlabel("Z (Å)")  # type: ignore[attr-defined]
            self.canvas.draw_idle()

        # Update Molara info if available
        if HAS_MOLARA:
            num_atoms = len(positions) if positions is not None else 0
            fname = filename or "In-memory structure"
            self.molara_info_label.setText(
                f"<h3>Molara 3D Live Renderer</h3>"
                f"<p>Loaded: <b>{fname}</b> ({num_atoms} atoms)</p>"
                f"<p>Click 'Launch Molara Live Viewer' to explore in 3D with full orbital, "
                f"density, and trajectory rendering support.</p>",
            )

    def _launch_molara(self) -> None:
        """Trigger Molara viewer on the currently loaded structure."""
        if not HAS_MOLARA:
            return

        try:
            import molara  # noqa: PLC0415

            # If molara provides an entry point or API:
            if hasattr(molara, "view") and self._current_filename:
                molara.view(self._current_filename)
            elif hasattr(molara, "main"):
                molara.main()
            else:
                self.status_label.setText(
                    "<b>3D View:</b> <span style='color: green;'>Molara invoked</span>",
                )
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(
                f"<b>3D View:</b> <span style='color: red;'>Molara error: {exc}</span>",
            )

    def clear(self) -> None:
        """Clear the current rendering."""
        self._current_positions = None
        self._current_species = None
        self._current_cell = None
        self._current_filename = None

        if self.ax is not None:
            self.ax.clear()
            self.ax.set_title("No structure loaded")
            self.ax.set_xlabel("X (Å)")
            self.ax.set_ylabel("Y (Å)")
            self.ax.set_zlabel("Z (Å)")  # type: ignore[attr-defined]
            self.canvas.draw_idle()

        if HAS_MOLARA:
            self.molara_info_label.setText(
                "<h3>Molara 3D Live Renderer</h3><p>No structure loaded. Open a file to begin.</p>",
            )
