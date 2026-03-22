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
    """

    title: str = "Training Loss"
    max_points: int = 200
    height: int = 15
    width: int | None = None

    # Internal storage
    _train_steps: deque[int] = field(default_factory=deque, repr=False)
    _train_losses: deque[float] = field(default_factory=deque, repr=False)
    _val_steps: deque[int] = field(default_factory=deque, repr=False)
    _val_losses: deque[float] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        # Initialize deques with maxlen for automatic truncation
        self._train_steps = deque(maxlen=self.max_points)
        self._train_losses = deque(maxlen=self.max_points)
        self._val_steps = deque(maxlen=self.max_points)
        self._val_losses = deque(maxlen=self.max_points)

    def add_train_loss(self, step: int, loss: float) -> None:
        """Record a training loss value."""
        self._train_steps.append(step)
        self._train_losses.append(loss)

    def add_val_loss(self, step: int, loss: float) -> None:
        """Record a validation loss value."""
        self._val_steps.append(step)
        self._val_losses.append(loss)

    def plot(self) -> None:
        """Render the loss plot to the terminal.

        Clears the previous plot and draws the current state.
        Should be called after adding new data points.
        """
        plt.clear_figure()
        plt.theme("dark")

        # Plot training loss
        if self._train_steps:
            plt.plot(
                list(self._train_steps),
                list(self._train_losses),
                label="train",
                marker="braille",
            )

        # Plot validation loss
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

        # Show legend only if we have data
        if self._train_steps or self._val_steps:
            plt.show()

    def clear(self) -> None:
        """Clear all recorded data."""
        self._train_steps.clear()
        self._train_losses.clear()
        self._val_steps.clear()
        self._val_losses.clear()

    def get_latest_train_loss(self) -> float | None:
        """Get the most recent training loss, or None if no data."""
        return self._train_losses[-1] if self._train_losses else None

    def get_latest_val_loss(self) -> float | None:
        """Get the most recent validation loss, or None if no data."""
        return self._val_losses[-1] if self._val_losses else None
