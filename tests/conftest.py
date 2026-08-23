"""Shared pytest configuration: register markers and set non-interactive backends."""

from __future__ import annotations

import os

import matplotlib as mpl

# Use a non-interactive matplotlib backend in test runs.
mpl.use("Agg")

# OVITO loads a Qt platform plugin on import. On headless CI machines
# (e.g. windows-latest runners without a display server) the default
# platform can crash natively (access violation). The "offscreen" Qt
# platform avoids this. It must be set before ovito is first imported,
# which is why it lives here rather than in the individual tests.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
