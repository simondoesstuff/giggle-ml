"""Terminal-based loss plotting using plotext.

Provides a reusable utility for plotting training metrics in the terminal
alongside tqdm progress bars.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import plotext as plt


@dataclass
class TerminalLossPlotter:
    """Plot loss curves in the terminal during training.

    Designed to work alongside tqdm progress bars by using plotext's
    terminal-aware plotting.

    Attributes:
        title: Plot title.
        max_points: Maximum points to display per series (older points dropped).
        height: Plot height in terminal rows.
        width: Plot width in terminal columns (None for auto).
        aux_metric_names: Names of auxiliary metrics to track (e.g., ["nDCG"]).
    """

    title: str = "Training Loss"
    max_points: int = 200
    height: int = 15
    width: int | None = None
    aux_metric_names: list[str] = field(default_factory=list)

    # Internal storage
    _train_steps: deque[int] = field(default_factory=deque, repr=False)
    _train_losses: deque[float] = field(default_factory=deque, repr=False)
    _val_steps: deque[int] = field(default_factory=deque, repr=False)
    _val_losses: deque[float] = field(default_factory=deque, repr=False)
    # Auxiliary metrics: name -> (steps deque, values deque)
    _aux_metrics: dict[str, tuple[deque[int], deque[float]]] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        # Initialize deques with maxlen for automatic truncation
        self._train_steps = deque(maxlen=self.max_points)
        self._train_losses = deque(maxlen=self.max_points)
        self._val_steps = deque(maxlen=self.max_points)
        self._val_losses = deque(maxlen=self.max_points)
        # Initialize aux metric storage
        self._aux_metrics = {
            name: (deque(maxlen=self.max_points), deque(maxlen=self.max_points))
            for name in self.aux_metric_names
        }

    def add_train_loss(self, step: int, loss: float) -> None:
        """Record a training loss value."""
        self._train_steps.append(step)
        self._train_losses.append(loss)

    def add_val_loss(self, step: int, loss: float) -> None:
        """Record a validation loss value."""
        self._val_steps.append(step)
        self._val_losses.append(loss)

    def add_aux_metric(self, name: str, step: int, value: float) -> None:
        """Record an auxiliary metric value.

        Args:
            name: Metric name (must be in aux_metric_names).
            step: Training step.
            value: Metric value.
        """
        if name not in self._aux_metrics:
            # Dynamically add if not pre-registered
            self._aux_metrics[name] = (
                deque(maxlen=self.max_points),
                deque(maxlen=self.max_points),
            )
        steps, values = self._aux_metrics[name]
        steps.append(step)
        values.append(value)

    def plot(self) -> None:
        """Render the loss plot to the terminal.

        Clears the previous plot and draws the current state.
        Should be called after adding new data points.

        If auxiliary metrics are present, uses subplots with loss on top
        and aux metrics below.
        """
        plt.clear_figure()
        plt.theme("dark")

        # Check if we have aux metrics with data
        aux_with_data = [
            (name, steps, values)
            for name, (steps, values) in self._aux_metrics.items()
            if steps
        ]

        if aux_with_data:
            # Use subplots: loss on top, each aux metric below
            n_plots = 1 + len(aux_with_data)
            plt.subplots(n_plots, 1)

            # Plot 1: Loss curves
            plt.subplot(1, 1)
            if self._train_steps:
                plt.plot(
                    list(self._train_steps),
                    list(self._train_losses),
                    label="train",
                    marker="braille",
                )
            if self._val_steps:
                plt.plot(
                    list(self._val_steps),
                    list(self._val_losses),
                    label="val",
                    marker="braille",
                )
            plt.title(self.title)
            plt.ylabel("Loss")

            # Plot aux metrics
            for i, (name, steps, values) in enumerate(aux_with_data, start=2):
                plt.subplot(i, 1)
                plt.plot(
                    list(steps),
                    list(values),
                    label=name,
                    marker="braille",
                )
                plt.ylabel(name)

            # Only show xlabel on bottom plot
            plt.xlabel("Step")
            plt.plotsize(self.width, self.height * n_plots)
        else:
            # Single plot for loss only
            if self._train_steps:
                plt.plot(
                    list(self._train_steps),
                    list(self._train_losses),
                    label="train",
                    marker="braille",
                )
            if self._val_steps:
                plt.plot(
                    list(self._val_steps),
                    list(self._val_losses),
                    label="val",
                    marker="braille",
                )
            plt.title(self.title)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.plotsize(self.width, self.height)

        # Show if we have any data
        if self._train_steps or self._val_steps or aux_with_data:
            plt.show()

    def clear(self) -> None:
        """Clear all recorded data."""
        self._train_steps.clear()
        self._train_losses.clear()
        self._val_steps.clear()
        self._val_losses.clear()
        for steps, values in self._aux_metrics.values():
            steps.clear()
            values.clear()

    def get_latest_train_loss(self) -> float | None:
        """Get the most recent training loss, or None if no data."""
        return self._train_losses[-1] if self._train_losses else None

    def get_latest_val_loss(self) -> float | None:
        """Get the most recent validation loss, or None if no data."""
        return self._val_losses[-1] if self._val_losses else None

    def get_latest_aux_metric(self, name: str) -> float | None:
        """Get the most recent value for an auxiliary metric, or None if no data."""
        if name not in self._aux_metrics:
            return None
        _, values = self._aux_metrics[name]
        return values[-1] if values else None
