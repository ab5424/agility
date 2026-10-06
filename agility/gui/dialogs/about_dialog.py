# Copyright (c) 2021 Alexander Bonkowski
# Distributed under the terms of the MIT License
# author: Alexander Bonkowski

"""About dialog for Agility."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
)

import agility

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget


class AboutDialog(QDialog):
    """About dialog providing version and backend detection information."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize About dialog."""
        super().__init__(parent)
        self.setWindowTitle("About Agility")
        self.setFixedSize(460, 380)

        layout = QVBoxLayout(self)

        title_label = QLabel(f"<h2>Agility v{agility.__version__}</h2>", self)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        desc_label = QLabel(
            "<p><b>Atomistic Grain Boundary and Interface Utility</b></p>"
            "<p>A Python library for pre- and postprocessing polycrystalline and "
            "grain-boundary structures for atomistic codes (LAMMPS, VASP, etc.).</p>"
            "<p>Copyright &copy; 2021 Alexander Bonkowski.<br>"
            "Distributed under the terms of the MIT License.</p>",
            self,
        )
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_label)

        # Optional dependencies status table
        backends_status = self._check_dependencies()
        status_html = "<h4>Detected Packages:</h4><ul>"
        for name, status in backends_status.items():
            status_html += f"<li><b>{name}:</b> {status}</li>"
        status_html += "</ul>"

        status_label = QLabel(status_html, self)
        status_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(status_label)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, self)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

    @staticmethod
    def _check_dependencies() -> dict[str, str]:
        """Detect which optional dependencies are installed."""
        packages = ["ovito", "ase", "pymatgen", "molara", "lammps", "matplotlib"]
        status: dict[str, str] = {}
        for pkg in packages:
            spec = find_spec(pkg)
            if spec is not None:
                status[pkg] = "<span style='color: green;'>Available</span>"
            else:
                status[pkg] = "<span style='color: gray;'>Not installed</span>"
        return status
