# Copyright (c) 2021 Alexander Bonkowski
# Distributed under the terms of the MIT License
# author: Alexander Bonkowski

"""Agility GUI application launcher and entrypoint."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from agility import __version__
from agility.gui.main_window import AgilityMainWindow

if TYPE_CHECKING:
    from collections.abc import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the Agility GUI.

    Args:
        argv: Command-line arguments sequence.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Agility - Atomistic Grain Boundary & Interface Analysis GUI",
    )
    parser.add_argument(
        "structure_file",
        nargs="?",
        default=None,
        help="Optional path to an atomistic structure file to open upon launch.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Agility {__version__}",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Agility GUI application.

    Args:
        argv: Optional list of command-line arguments.

    Returns:
        Application exit code.
    """
    args = parse_args(argv)

    # Enable High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough,
    )

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv if argv is None else [sys.argv[0], *list(argv)])

    app.setApplicationName("Agility")
    app.setOrganizationName("Agility")
    app.setApplicationVersion(__version__)

    main_win = AgilityMainWindow()
    main_win.show()

    if args.structure_file:
        main_win.load_structure_file(args.structure_file)

    # If running in an interactive session or unit test, app.exec() may not be called
    if not hasattr(app, "_agility_test_mode"):
        return app.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main())
