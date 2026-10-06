# Copyright (c) 2021 Alexander Bonkowski
# Distributed under the terms of the MIT License
# author: Alexander Bonkowski

"""Background worker threads for Agility GUI operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, QThread, Signal

if TYPE_CHECKING:
    from collections.abc import Callable


class WorkerSignals(QObject):
    """Signals for background execution."""

    started = Signal()
    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(str)


class AnalysisWorker(QThread):
    """Worker thread for running Agility computational tasks in the background."""

    def __init__(
        self,
        task_func: Callable[..., Any],
        *args: object,
        **kwargs: object,
    ) -> None:
        """Initialize the worker thread.

        Args:
            task_func: The callable to execute.
            *args: Positional arguments for task_func.
            **kwargs: Keyword arguments for task_func.
        """
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        """Execute the task callable."""
        self.signals.started.emit()
        try:
            result = self.task_func(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))
