"""Shared pytest configuration: register markers and set non-interactive backends."""

from __future__ import annotations

import os

import matplotlib as mpl

# Use a non-interactive matplotlib backend in test runs.
mpl.use("Agg")

# OVITO loads a Qt platform plugin and initialises its graphics stack on
# import. On headless CI machines (e.g. windows-latest runners without a
# display server or GPU) this can crash natively (access violation). The
# "offscreen" Qt platform plus software rendering avoid this. They must be
# set before ovito is first imported, which is why they live here rather
# than in the individual tests.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OVITO_GL_RENDERER", "software")
