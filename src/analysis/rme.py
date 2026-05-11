"""Dense heatmap visualization and RME similarity parsing."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.plots.dense_heat_matrix import plot_dense_heatmap
from giggleml.utils.file_utils import Pathish
from giggleml.utils.math import sigmoid_with_temperature

# ==========================================
# Genomics Domain Specifics
# ==========================================

chromatin_states = [
    "Active_TSS",
    "Flanking_Active_TSS",
    "Strong_transcription",
    "Weak_transcription",
    "Enhancers",
    "Genic_enhancers",
    "ZNF_genes_and_repeats",
    "Heterochromatin",
    "Bivalent_Poised_TSS",
    "Flanking_Bivalent_TSS_Enh",
    "Bivalent_Enhancer",
    "Repressed_PolyComb",
    "Weak_Repressed_PolyComb",
    "Transcr_at_gene_5_and_3",
    "Quiescent_Low",
]

# Updated categories as requested
cell_categories = [
    "iPSC",
    "Thymus",
    "Sm muscle",
    "Other",
    "Neurosph",
    "Muscle",
    "Mesench",
    "Lung",
    "Heart",
    "HSC and B cell",
    "Epithelial",
    "ESC",
    "ES deriv",
    "Digestive",
    "Cancer cell line",
    "Brain",
    "Blood and T-cell",
]

# Broader category groupings
broad_categories = [
    "Pluripotent",
    "Blood/Immune",
    "Muscle/Heart",
    "Neural",
    "Stromal",
    "Organ",
    "Cancer",
    "Other",
]

category_to_broad_map = {
    "iPSC": "Pluripotent",
    "ESC": "Pluripotent",
    "ES deriv": "Pluripotent",
    "Thymus": "Blood/Immune",
    "HSC and B cell": "Blood/Immune",
    "Blood and T-cell": "Blood/Immune",
    "Sm muscle": "Muscle/Heart",
    "Muscle": "Muscle/Heart",
    "Heart": "Muscle/Heart",
    "Neurosph": "Neural",
    "Brain": "Neural",
    "Mesench": "Stromal",
    "Epithelial": "Stromal",
    "Lung": "Organ",
    "Digestive": "Organ",
    "Cancer cell line": "Cancer",
    "Other": "Other",
}


def classify_broad_category(category: str) -> str:
    """Convert a cell category to its broader category grouping."""
    return category_to_broad_map.get(category, "Other")


category_keywords_map = {
    "iPSC": ["IPS"],
    "Thymus": ["THYMUS", "SPLEEN"],
    "Sm muscle": ["SMOOTH_MUSCLE"],
    "Neurosph": ["NEUROSPHERE"],
    "Muscle": ["SKELETAL", "HSMM", "PSOAS", "FETAL_MUSCLE"],
    "Mesench": [
        "MESENCHYMAL",
        "FIBROBLAST",
        "ADIPOSE",
        "OSTEOBLAST",
        "CHONDROCYTE",
        "IMR90",
    ],
    "Lung": ["LUNG", "NHLF"],
    "Heart": ["HEART", "AORTA", "VENTRICLE", "ATRIUM", "HUVEC"],
    "HSC and B cell": ["CD19", "CD34", "GM12878"],
    "Epithelial": ["EPITHELIAL", "HMEC", "BREAST"],
    "ESC": ["H1_CELL_LINE", "H9_CELL_LINE", "HUES", "ES_WA7", "ES_I3"],
    "ES deriv": ["H1_BMP4_DERIVED", "HESC_DERIVED", "H1_DERIVED", "H9_DERIVED"],
    "Digestive": [
        "LIVER",
        "STOMACH",
        "INTESTINE",
        "COLON",
        "RECTAL",
        "DUODENUM",
        "ESOPHAGUS",
        "GASTRIC",
        "MUCOSA",
    ],
    "Cancer cell line": [
        "CARCINOMA",
        "LEUKEMIA",
        "K562",
        "HEPG2",
        "HELA",
        "A549",
        "DND41",
    ],
    "Brain": ["BRAIN", "ASTROCYTE"],
    "Blood and T-cell": [
        "CD3",
        "CD4",
        "CD8",
        "CD14",
        "CD15",
        "CD56",
        "PERIPHERAL_BLOOD",
        "MONOCYTE",
    ],
    "Other": [],  # Used as a fallback
}


def classify_cell_type(cell_type: str) -> str:
    cell_type_upper = cell_type.upper()
    for category, keywords in category_keywords_map.items():
        if any(keyword in cell_type_upper for keyword in keywords):
            return category
    return "Other"  # Default to 'Other' instead of raising error


def classify_bed_file(bed: Pathish) -> tuple[str, str]:
    filename = os.path.basename(str(bed))

    # Strip extensions
    if filename.endswith(".bed.gz"):
        filename = filename[:-7]
    elif filename.endswith(".bed"):
        filename = filename[:-4]

    # Look for matching chromatin state suffix
    for state in sorted(chromatin_states, key=len, reverse=True):
        if filename.endswith(state):
            prefix = filename[: -(len(state))].rstrip("_")
            return prefix, state

    raise ValueError(f'Unknown format, expecting "cellType_chrmState", got: {filename}')


def parse_score_file(score_path: Pathish) -> tuple[list[str], list[float]]:
    def parse() -> Iterator[tuple[str, float]]:
        with open(score_path, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                split = line.strip().split()
                yield Path(split[0]).name.split(".")[0], float(split[-1])

    # Ensure list types are returned to match type hints
    parsed_tuples = sorted(parse(), key=lambda x: -x[1])
    if not parsed_tuples:
        return [], []

    keys, values = zip(*parsed_tuples)
    return list(keys), list(values)


@dataclass
class _PlotData:
    """Preprocessed data for a single heatmap."""

    name: str
    data: np.ndarray
    ordered_cells: list[str]
    bed_to_category: dict[str, str]


def _prepare_plot_data(
    score_paths: tuple[Pathish, ...],
    names: tuple[str, ...],
    states: list[str] | None = None,
) -> list[_PlotData]:
    """Parse score files and prepare data for plotting."""
    col_labels = states if states is not None else chromatin_states
    n_cols = len(col_labels)
    plot_data_list: list[_PlotData] = []

    for path, name in zip(score_paths, names):
        beds, scores = parse_score_file(path)

        cell_types: set[str] = set()
        parsed_data: dict[tuple[str, str], float] = {}

        for bed, score in zip(beds, scores):
            try:
                cell_id, state = classify_bed_file(bed)
                cell_types.add(cell_id)
                parsed_data[(cell_id, state)] = score
            except ValueError:
                continue

        if not cell_types:
            print(
                f"Warning: No valid parsed data found for {name}. Skipping.",
                file=sys.stderr,
            )
            continue

        ordered_cells = sorted(list(cell_types))
        n_rows = len(ordered_cells)

        data = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
        bed_to_category: dict[str, str] = {}

        for i, cell_id in enumerate(ordered_cells):
            try:
                cat = classify_cell_type(cell_id)
            except ValueError:
                cat = "Other"

            bed_to_category[cell_id] = cat

            for j, state in enumerate(col_labels):
                data[i, j] = parsed_data.get((cell_id, state), np.nan)

        plot_data_list.append(
            _PlotData(
                name=name,
                data=data,
                ordered_cells=ordered_cells,
                bed_to_category=bed_to_category,
            )
        )

    return plot_data_list


@dataclass
class SigmoidParams:
    """Parameters for sigmoid transformation."""

    temperature: float
    midpoint: float = 0.0


def plot_rme_similarity(
    score_paths: tuple[Pathish, ...],
    names: tuple[str, ...],
    output_path: Pathish | None = None,
    show: bool = True,
    sigmoid_params: tuple[SigmoidParams, ...] | None = None,
    states: list[str] | None = None,
    znorm: bool = False,
    cmap: str = "RdBu_r",
) -> None:
    if len(score_paths) != len(names):
        raise ValueError("Must provide the same number of score paths and names.")

    plot_data_list = _prepare_plot_data(score_paths, names, states=states)

    if not plot_data_list:
        print("No valid data to plot.", file=sys.stderr)
        return

    n_plots = len(plot_data_list)

    # Validate and expand sigmoid params
    if sigmoid_params is None:
        params: tuple[SigmoidParams | None, ...] = tuple([None] * n_plots)
    elif len(sigmoid_params) == 1:
        params = tuple([sigmoid_params[0]] * n_plots)
    elif len(sigmoid_params) == n_plots:
        params = sigmoid_params
    else:
        raise ValueError(
            f"sigmoid_params must have 1 or {n_plots} values, got {len(sigmoid_params)}"
        )

    col_labels = states if states is not None else chromatin_states
    cat_order = cell_categories

    # Compute figure dimensions
    max_rows = max(len(pd.ordered_cells) for pd in plot_data_list)
    subplot_width = 8
    subplot_height = max(3.0, max_rows * 0.1)
    fig_width = subplot_width * n_plots
    fig_height = subplot_height

    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(fig_width, fig_height),
        squeeze=False,
        layout="constrained",
    )

    for idx, pd in enumerate(plot_data_list):
        ax = axes[0, idx]
        sp = params[idx]

        # Apply sigmoid with temperature if specified
        if sp is not None:
            plot_data = sigmoid_with_temperature(pd.data, sp.temperature, sp.midpoint)
            title = f"{pd.name} (T={sp.temperature}, mid={sp.midpoint})"
            cbar_label = "Sigmoid Score"
        else:
            plot_data = pd.data
            title = pd.name
            cbar_label = "Similarity Score"

        if znorm:
            col_mean = np.nanmean(plot_data, axis=0, keepdims=True)
            col_std = np.nanstd(plot_data, axis=0, keepdims=True)
            plot_data = (plot_data - col_mean) / np.where(col_std == 0, 1.0, col_std)
            cbar_label = f"Z-score ({cbar_label})"

        def make_row_cat_fn(
            bed_to_cat: dict[str, str],
        ) -> Callable[[str], str]:
            def row_cat_fn(label: str) -> str:
                return bed_to_cat.get(label, "Other")

            return row_cat_fn

        plot_dense_heatmap(
            data=plot_data,
            row_labels=pd.ordered_cells,
            col_labels=col_labels,
            row_category_fn=make_row_cat_fn(pd.bed_to_category),
            row_category_order=cat_order,
            title=title,
            cmap=cmap,
            line_color="black",
            cbar_label=cbar_label,
            fig=fig,
            ax=ax,
        )

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {output_path}", file=sys.stderr)

    if show:
        plt.show()


# ==========================================
# CLI Entrypoint
# ==========================================


def main():
    parser = argparse.ArgumentParser(
        description="Plot comparative dense heatmaps from multiple RME score files."
    )
    parser.add_argument(
        "-s",
        "--scores",
        nargs="+",
        required=True,
        help="Paths to the score files (.score or .txt).",
    )
    parser.add_argument(
        "-n",
        "--names",
        nargs="+",
        required=True,
        help="Names corresponding to the score files, to be used as column labels.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (e.g., plot.png, plot.pdf). Saves with tight bounding box.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot interactively (useful with --output).",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        nargs="+",
        type=float,
        help="Sigmoid temperature(s). Lower = sharper. One value for all, or one per score file.",
    )
    parser.add_argument(
        "-m",
        "--midpoint",
        nargs="+",
        type=float,
        default=[0.0],
        help="Sigmoid midpoint(s) where output is 0.5. Default: 0.0.",
    )
    parser.add_argument(
        "--cmap",
        default="RdBu_r",
        help="Matplotlib colormap name (default: RdBu_r).",
    )
    parser.add_argument(
        "--znorm",
        action="store_true",
        help="Z-score normalize each chromatin state column across cell types before plotting.",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        choices=chromatin_states,
        metavar="STATE",
        help=(
            "Whitelist of chromatin states to include (default: all). "
            f"Choices: {', '.join(chromatin_states)}"
        ),
    )

    args = parser.parse_args()

    if len(args.scores) != len(args.names):
        print(
            "Error: The number of score paths (-s) must match the number of names (-n).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build sigmoid params if temperature is specified
    sigmoid_params = None
    if args.temperature:
        n = len(args.scores)

        def expand_param(values: list[float], name: str) -> list[float]:
            if len(values) == 1:
                return values * n
            elif len(values) == n:
                return values
            else:
                print(
                    f"Error: {name} must have 1 or {n} values, got {len(values)}.",
                    file=sys.stderr,
                )
                sys.exit(1)

        temps = expand_param(args.temperature, "--temperature")
        mids = expand_param(args.midpoint, "--midpoint")
        sigmoid_params = tuple(
            SigmoidParams(temperature=t, midpoint=m) for t, m in zip(temps, mids)
        )

    plot_rme_similarity(
        tuple(args.scores),
        tuple(args.names),
        output_path=args.output,
        show=not args.no_show,
        sigmoid_params=sigmoid_params,
        states=args.states,
        znorm=args.znorm,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
