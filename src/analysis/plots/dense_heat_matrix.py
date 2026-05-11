"""Dense heatmap visualization with optional category grouping.

Provides heatmaps for matrices with many labels, where individual labels
can be grouped into categories. Category boundaries are shown with dotted
lines, and axis labels display category names instead of individual labels
when grouping is enabled.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from numpy.typing import NDArray

# ==========================================
# Heatmap Visualization Module
# ==========================================


@dataclass
class CategoryMapping:
    """Maps individual labels to categories for axis grouping."""

    label_to_category: Callable[[str], str]
    category_order: Sequence[str] | None = None

    def get_category(self, label: str) -> str:
        return self.label_to_category(label)


@dataclass
class AxisConfig:
    """Configuration for a single axis of the heatmap."""

    labels: Sequence[str]
    category_mapping: CategoryMapping | None = None
    show_label_key: bool = False


@dataclass
class DenseHeatMatrix:
    """Dense heatmap with optional category grouping."""

    data: NDArray[np.floating]
    row_config: AxisConfig
    col_config: AxisConfig
    title: str | None = None
    cmap: str | Colormap = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    figsize: tuple[float, float] = (12, 10)
    cbar_label: str = "Value"
    show_colorbar: bool = True
    cbar_fraction: float = 0.05
    cbar_pad: float = 0.02
    line_color: str | None = None
    line_style: str = ":"
    line_width: float = 1.5

    _row_order: list[int] = field(default_factory=list, repr=False)
    _col_order: list[int] = field(default_factory=list, repr=False)
    _row_category_breaks: list[int] = field(default_factory=list, repr=False)
    _col_category_breaks: list[int] = field(default_factory=list, repr=False)
    _row_categories: list[str] = field(default_factory=list, repr=False)
    _col_categories: list[str] = field(default_factory=list, repr=False)

    def _compute_ordering_and_breaks(
        self,
        config: AxisConfig,
    ) -> tuple[list[int], list[int], list[str]]:
        labels = list(config.labels)
        if config.category_mapping is None:
            return list(range(len(labels))), [], []

        mapping = config.category_mapping
        category_to_indices: dict[str, list[int]] = {}
        for i, label in enumerate(labels):
            cat = mapping.get_category(label)
            if cat not in category_to_indices:
                category_to_indices[cat] = []
            category_to_indices[cat].append(i)

        if mapping.category_order is not None:
            categories = [c for c in mapping.category_order if c in category_to_indices]
            for cat in category_to_indices:
                if cat not in categories:
                    categories.append(cat)
        else:
            categories = sorted(category_to_indices.keys())

        ordered_indices: list[int] = []
        breaks: list[int] = []

        for cat in categories:
            if ordered_indices:
                breaks.append(len(ordered_indices))
            ordered_indices.extend(category_to_indices[cat])

        return ordered_indices, breaks, categories

    def _reorder_data(self) -> NDArray[np.floating]:
        return self.data[np.ix_(self._row_order, self._col_order)]

    def _compute_tick_positions(
        self,
        n_items: int,
        breaks: list[int],
        categories: list[str],
    ) -> tuple[list[float], list[str]]:
        if not categories:
            return list(range(n_items)), []

        positions: list[float] = []
        labels: list[str] = []
        boundaries = [0] + breaks + [n_items]

        for i, cat in enumerate(categories):
            start = boundaries[i]
            end = boundaries[i + 1]
            center = (start + end - 1) / 2
            positions.append(center)
            labels.append(cat)

        return positions, labels

    def _get_line_color(self, ax: Axes) -> str:
        if self.line_color is not None:
            return self.line_color
        from matplotlib.colors import to_rgba

        bg = to_rgba(ax.get_facecolor())
        luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        return "white" if luminance < 0.5 else "black"

    def _draw_category_lines(self, ax: Axes) -> None:
        line_kwargs = {
            "color": self._get_line_color(ax),
            "linestyle": self.line_style,
            "linewidth": self.line_width,
        }
        for brk in self._row_category_breaks:
            ax.axhline(y=brk - 0.5, xmin=0, xmax=1, **line_kwargs)
        for brk in self._col_category_breaks:
            ax.axvline(x=brk - 0.5, ymin=0, ymax=1, **line_kwargs)

    def plot(
        self, fig: Figure | None = None, ax: Axes | None = None
    ) -> tuple[Figure, Axes]:
        self._row_order, self._row_category_breaks, self._row_categories = (
            self._compute_ordering_and_breaks(self.row_config)
        )
        self._col_order, self._col_category_breaks, self._col_categories = (
            self._compute_ordering_and_breaks(self.col_config)
        )
        plot_data = self._reorder_data()

        subplot_mode = fig is not None and ax is not None
        show_row_key = (
            not subplot_mode
            and self.row_config.show_label_key
            and self.row_config.category_mapping
        )
        show_col_key = (
            not subplot_mode
            and self.col_config.show_label_key
            and self.col_config.category_mapping
        )

        if subplot_mode:
            assert fig is not None and ax is not None
        elif show_row_key or show_col_key:
            # Use GridSpec so the label-key panel doesn't fight with the heatmap
            key_width = 4.0
            fig = plt.figure(
                figsize=(self.figsize[0] + key_width, self.figsize[1]),
                layout="constrained",
            )
            gs = fig.add_gridspec(1, 2, width_ratios=[self.figsize[0], key_width])
            ax = fig.add_subplot(gs[0])
        else:
            fig, ax = plt.subplots(figsize=self.figsize, layout="constrained")

        vmin = self.vmin if self.vmin is not None else float(np.nanmin(self.data))
        vmax = self.vmax if self.vmax is not None else float(np.nanmax(self.data))

        cmap = plt.get_cmap(self.cmap) if isinstance(self.cmap, str) else self.cmap
        cmap = cmap.copy()
        cmap.set_bad(color="white", alpha=0)

        im = ax.imshow(
            plot_data,
            aspect="auto",
            cmap=cmap,
            norm=Normalize(vmin=vmin, vmax=vmax),
            interpolation="nearest",
        )

        self._draw_category_lines(ax)

        if self.row_config.category_mapping:
            tick_pos, tick_labels = self._compute_tick_positions(
                len(self._row_order), self._row_category_breaks, self._row_categories
            )
            ax.set_yticks(tick_pos)
            ax.set_yticklabels(tick_labels)
        else:
            ax.set_yticks(range(len(self.row_config.labels)))
            ax.set_yticklabels([self.row_config.labels[i] for i in self._row_order])

        if self.col_config.category_mapping:
            tick_pos, tick_labels = self._compute_tick_positions(
                len(self._col_order), self._col_category_breaks, self._col_categories
            )
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        else:
            ax.set_xticks(range(len(self.col_config.labels)))
            ax.set_xticklabels(
                [self.col_config.labels[i] for i in self._col_order],
                rotation=45,
                ha="right",
            )

        if self.title:
            ax.set_title(self.title, fontsize=14, fontweight="bold")

        if self.show_colorbar:
            cbar = fig.colorbar(
                im, ax=ax, fraction=self.cbar_fraction, pad=self.cbar_pad
            )
            cbar.set_label(self.cbar_label)

        if show_row_key or show_col_key:
            assert fig is not None
            self._add_label_keys(fig, bool(show_row_key), bool(show_col_key))

        return fig, ax

    def _add_label_keys(
        self, fig: Figure, show_row_key: bool, show_col_key: bool
    ) -> None:
        text_color = plt.rcParams["text.color"]
        bg_color = plt.rcParams["axes.facecolor"]
        bbox_style = {
            "boxstyle": "round",
            "facecolor": bg_color,
            "alpha": 0.8,
            "edgecolor": text_color,
        }

        def build_category_blocks(
            categories: list[str],
            order: list[int],
            labels: Sequence[str],
            mapping: CategoryMapping,
        ) -> list[str]:
            blocks: list[str] = []
            for cat in categories:
                cat_indices = [
                    i for i in order if mapping.get_category(labels[i]) == cat
                ]
                lines = [f"{cat}:"]
                for i in cat_indices:
                    lines.append(f"  {i + 1}. {labels[i]}")
                blocks.append("\n".join(lines))
            return blocks

        all_blocks: list[str] = []
        if show_row_key and self.row_config.category_mapping:
            all_blocks.extend(
                build_category_blocks(
                    self._row_categories,
                    self._row_order,
                    list(self.row_config.labels),
                    self.row_config.category_mapping,
                )
            )
        if show_col_key and self.col_config.category_mapping:
            all_blocks.extend(
                build_category_blocks(
                    self._col_categories,
                    self._col_order,
                    list(self.col_config.labels),
                    self.col_config.category_mapping,
                )
            )

        n_blocks = len(all_blocks)
        n_cols = min(n_blocks, 2 if n_blocks <= 4 else 3)
        col_width = 0.22 / n_cols * 2

        for idx, block in enumerate(all_blocks):
            col = idx % n_cols
            row = idx // n_cols
            n_rows = (n_blocks + n_cols - 1) // n_cols
            x = 0.76 + col * col_width
            y = 0.85 - row * (0.7 / max(n_rows, 1))

            fig.text(
                x,
                y,
                block,
                fontsize=7,
                fontfamily="monospace",
                verticalalignment="top",
                color=text_color,
                bbox=bbox_style,
            )

    def save(
        self, path: str, dpi: int = 150, fmt: Literal["png", "pdf", "svg"] | None = None
    ) -> None:
        fig, _ = self.plot()
        fig.savefig(path, dpi=dpi, format=fmt, bbox_inches="tight")
        plt.close(fig)


def plot_dense_heatmap(
    data: NDArray[np.floating],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    row_category_fn: Callable[[str], str] | None = None,
    col_category_fn: Callable[[str], str] | None = None,
    row_category_order: Sequence[str] | None = None,
    col_category_order: Sequence[str] | None = None,
    show_row_key: bool = False,
    show_col_key: bool = False,
    title: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "Value",
    show_colorbar: bool = True,
    cbar_fraction: float = 0.05,
    cbar_pad: float = 0.02,
    figsize: tuple[float, float] = (12, 10),
    line_color: str | None = None,
    line_style: str = ":",
    line_width: float = 1.5,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    row_mapping = (
        CategoryMapping(row_category_fn, row_category_order)
        if row_category_fn
        else None
    )
    col_mapping = (
        CategoryMapping(col_category_fn, col_category_order)
        if col_category_fn
        else None
    )

    row_config = AxisConfig(
        labels=row_labels, category_mapping=row_mapping, show_label_key=show_row_key
    )
    col_config = AxisConfig(
        labels=col_labels, category_mapping=col_mapping, show_label_key=show_col_key
    )

    heatmap = DenseHeatMatrix(
        data=data,
        row_config=row_config,
        col_config=col_config,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_label=cbar_label,
        show_colorbar=show_colorbar,
        cbar_fraction=cbar_fraction,
        cbar_pad=cbar_pad,
        figsize=figsize,
        line_color=line_color,
        line_style=line_style,
        line_width=line_width,
    )
    return heatmap.plot(fig=fig, ax=ax)


if __name__ == "__main__":
    np.random.seed(42)

    n_rows, n_cols = 80, 6
    data = np.random.randn(n_rows, n_cols)

    row_labels = [f"feature_{i:03d}" for i in range(n_rows)]
    col_labels = [f"sample_{i:02d}" for i in range(n_cols)]

    def row_to_category(label: str) -> str:
        idx = int(label.split("_")[1])
        categories = ["TypeA", "TypeB", "TypeC", "TypeD"]
        return categories[idx % len(categories)]

    def col_to_category(label: str) -> str:
        idx = int(label.split("_")[1])
        return "Control" if idx < 20 else "Treatment"

    plt.style.use("dark_background")
    fig, ax = plot_dense_heatmap(
        data,
        row_labels,
        col_labels,
        row_category_fn=row_to_category,
        # col_category_fn=col_to_category,
        row_category_order=["TypeA", "TypeB", "TypeC", "TypeD"],
        col_category_order=["Control", "Treatment"],
        show_row_key=True,
        title="Dense Heatmap with Category Grouping",
        cmap="RdBu_r",
        line_color="black",
        cbar_label="Z-score",
        figsize=(4, 10),
    )
    plt.show()
