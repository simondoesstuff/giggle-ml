"""Keyboard-navigable viewer for a series of plots.

Use left/right arrow keys to scroll through images.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any


def natural_sort_key(path: Path) -> list[int | str]:
    """Key function for natural sorting (handles numeric parts correctly)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event
from matplotlib.figure import Figure
from matplotlib.image import AxesImage


class PlotViewer:
    """Interactive viewer for scrolling through a series of images."""

    paths: list[Path]
    index: int
    fig: Figure
    ax: Axes
    im: AxesImage | None

    def __init__(self, image_paths: list[Path]) -> None:
        self.paths = image_paths
        self.index = 0

        plt.style.use("dark_background")
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.im = None
        self._update()

    def _update(self) -> None:
        """Update the displayed image."""
        img = plt.imread(self.paths[self.index])

        if self.im is None:
            self.im = self.ax.imshow(img)
            self.ax.axis("off")
        else:
            self.im.set_data(img)

        title = f"[{self.index + 1}/{len(self.paths)}] {self.paths[self.index].name}"
        self.ax.set_title(title, fontsize=10)
        self.fig.canvas.draw_idle()

    def _on_key(self, event: Event) -> Any:
        """Handle keyboard events."""
        key = getattr(event, "key", None)
        if key == "right":
            self.index = (self.index + 1) % len(self.paths)
            self._update()
        elif key == "left":
            self.index = (self.index - 1) % len(self.paths)
            self._update()
        elif key in ("q", "escape"):
            plt.close(self.fig)

    def show(self) -> None:
        """Display the viewer."""
        plt.show()


def collect_images(paths: list[Path]) -> list[Path]:
    """Collect image files from paths (files or directories)."""
    extensions = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
    images: list[Path] = []

    for p in paths:
        if p.is_dir():
            for ext in extensions:
                images.extend(p.glob(f"*{ext}"))
        elif p.is_file() and p.suffix.lower() in extensions:
            images.append(p)

    return sorted(images, key=natural_sort_key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scroll through plots with arrow keys.",
        epilog="Controls: LEFT/RIGHT = navigate, Q/ESC = quit",
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="Image files or directories containing images.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["name", "mtime"],
        default="name",
        help="Sort order (default: name).",
    )

    args = parser.parse_args()

    images = collect_images(args.paths)

    if not images:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    if args.sort_by == "mtime":
        images.sort(key=lambda p: p.stat().st_mtime)

    print(
        f"Found {len(images)} images. Use LEFT/RIGHT to navigate, Q to quit.",
        file=sys.stderr,
    )

    viewer = PlotViewer(images)
    viewer.show()


if __name__ == "__main__":
    main()
