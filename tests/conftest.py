"""Shared pytest configuration: register markers and set matplotlib to a non-interactive backend."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
