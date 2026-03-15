"""Unit tests for the minimise_lmp function (no real LAMMPS required)."""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import MagicMock

import pytest

from agility.minimiser import minimise_lmp


@pytest.mark.unit
class TestMinimiseLmp(TestCase):
    """Test the minimise_lmp function using mock LAMMPS objects."""

    def test_wrong_min_opt_length_raises_value_error(self) -> None:
        """Test that a min_opt with wrong length raises ValueError."""
        mock_lmp = MagicMock()
        with pytest.raises(ValueError, match="four arguments"):
            minimise_lmp(mock_lmp, min_opt=(0, 1e-8, 1000))

    def test_minimise_lmp_calls_correct_methods(self) -> None:
        """Test that minimise_lmp calls the expected LAMMPS methods with default args."""
        mock_lmp = MagicMock()
        result = minimise_lmp(mock_lmp)
        mock_lmp.min_style.assert_called_once_with("fire")
        # The four default min_opt values are formatted into a single space-separated string
        default_min_opt = (0, 1e-8, 1000, 100000)
        mock_lmp.minimize.assert_called_once_with(
            f"{default_min_opt[0]} {default_min_opt[1]} {default_min_opt[2]} {default_min_opt[3]}",
        )
        assert result is mock_lmp

    def test_minimise_lmp_custom_style(self) -> None:
        """Test that minimise_lmp forwards the requested minimization style."""
        mock_lmp = MagicMock()
        minimise_lmp(mock_lmp, style="cg")
        mock_lmp.min_style.assert_called_once_with("cg")

    def test_minimise_lmp_with_mod(self) -> None:
        """Test that minimise_lmp applies min_modify commands when mod is given."""
        mock_lmp = MagicMock()
        minimise_lmp(mock_lmp, mod=[("line", "quadratic")])
        mock_lmp.min_modify.assert_called_once_with("line quadratic")
