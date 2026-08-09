"""Unit tests for minimiser.py (no real LAMMPS required)."""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import MagicMock

import pytest

from agility.minimiser import minimise_lmp


@pytest.mark.unit
class TestMinimiseLmp(TestCase):
    """Test the ``minimise_lmp`` function using mock LAMMPS objects."""

    def test_wrong_min_opt_length_raises_value_error(self) -> None:
        """A ``min_opt`` with wrong length must raise ``ValueError``."""
        mock_lmp = MagicMock()
        with pytest.raises(ValueError, match="four arguments"):
            minimise_lmp(mock_lmp, min_opt=(0, 1e-8, 1000))

    def test_minimise_lmp_calls_correct_methods(self) -> None:
        """``minimise_lmp`` must call the expected LAMMPS methods with default args."""
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
        """``minimise_lmp`` must forward the requested minimisation style."""
        mock_lmp = MagicMock()
        minimise_lmp(mock_lmp, style="cg")
        mock_lmp.min_style.assert_called_once_with("cg")

    def test_minimise_lmp_with_mod(self) -> None:
        """``minimise_lmp`` must apply ``min_modify`` commands when ``mod`` is given."""
        mock_lmp = MagicMock()
        minimise_lmp(mock_lmp, mod=[("line", "quadratic")])
        mock_lmp.min_modify.assert_called_once_with("line quadratic")

    def test_minimise_lmp_custom_min_opt(self) -> None:
        """``minimise_lmp`` must forward custom ``min_opt`` values to ``minimize``."""
        mock_lmp = MagicMock()
        custom_opt = (1e-6, 1e-6, 500, 50000)
        minimise_lmp(mock_lmp, min_opt=custom_opt)
        mock_lmp.minimize.assert_called_once_with(
            f"{custom_opt[0]} {custom_opt[1]} {custom_opt[2]} {custom_opt[3]}",
        )

    def test_minimise_lmp_no_mod_does_not_call_min_modify(self) -> None:
        """When ``mod`` is ``None``, ``min_modify`` must not be called."""
        mock_lmp = MagicMock()
        minimise_lmp(mock_lmp, mod=None)
        mock_lmp.min_modify.assert_not_called()

    def test_minimise_lmp_multiple_mods(self) -> None:
        """Multiple ``mod`` entries must each produce a ``min_modify`` call."""
        mock_lmp = MagicMock()
        minimise_lmp(mock_lmp, mod=[("line", "quadratic"), ("norm", "2.0")])
        assert mock_lmp.min_modify.call_count == 2
        mock_lmp.min_modify.assert_any_call("line quadratic")
        mock_lmp.min_modify.assert_any_call("norm 2.0")

    def test_minimise_lmp_returns_lmp(self) -> None:
        """``minimise_lmp`` must return the same LAMMPS object it was given."""
        mock_lmp = MagicMock()
        result = minimise_lmp(mock_lmp)
        assert result is mock_lmp

    def test_minimise_lmp_empty_min_opt_raises(self) -> None:
        """An empty ``min_opt`` must raise ``ValueError``."""
        mock_lmp = MagicMock()
        with pytest.raises(ValueError, match="four arguments"):
            minimise_lmp(mock_lmp, min_opt=())

    def test_minimise_lmp_five_element_min_opt_raises(self) -> None:
        """A five-element ``min_opt`` must raise ``ValueError``."""
        mock_lmp = MagicMock()
        with pytest.raises(ValueError, match="four arguments"):
            minimise_lmp(mock_lmp, min_opt=(0, 1e-8, 1000, 100000, 999))
