"""UMAP visualization of Roadmap Epigenomics embeddings.

Produces side-by-side UMAP plots colored by cell category and chromatin state.
Supports both direct embeddings (zarr) and similarity matrices.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
import zarr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from analysis.rme import (
    broad_categories,
    chromatin_states,
    classify_bed_file,
    classify_broad_category,
    classify_cell_type,
)
from giggleml.data.similarity_matrix import SimilarityMatrix


def load_embeddings(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load embeddings from zarr.

    Returns:
        Tuple of (embeddings as numpy array, list of bed names).
    """
    z = zarr.open_array(str(path), mode="r")
    embeddings = np.asarray(z[:], dtype=np.float32)

    raw_names = z.attrs.get("bed_names", [])
    bed_names: list[str] = (
        [str(n) for n in raw_names] if isinstance(raw_names, list) else []
    )
    return embeddings, bed_names


def load_similarity_matrix(
    matrix_path: Path, names_file: Path
) -> tuple[np.ndarray, list[str]]:
    """Load a similarity matrix and convert to distance matrix for UMAP.

    Args:
        matrix_path: Path to the similarity matrix file (.mat).
        names_file: File containing BED names (one per line, will be sorted).

    Returns:
        Tuple of (distance matrix as numpy array, list of bed names in sorted order).
    """
    # Read and sort names (same order as used during matrix construction)
    with open(names_file) as f:
        bed_names = sorted(line.strip() for line in f if line.strip())
    n = len(bed_names)

    if n == 0:
        raise ValueError(f"No names found in {names_file}")

    # Load the similarity matrix
    matrix = SimilarityMatrix(matrix_path, n, mode="r")
    similarity = np.asarray(matrix.array, dtype=np.float32)

    # Convert similarity to distance
    # combo_score can be negative, so we shift and invert:
    # distance = max(similarity) - similarity
    # This ensures the most similar pairs have distance 0
    max_sim = np.max(similarity)
    distance = max_sim - similarity

    # Ensure diagonal is 0 (self-distance)
    np.fill_diagonal(distance, 0)

    return distance, bed_names


def parse_bed_labels(
    bed_names: list[str],
) -> tuple[list[str], list[str], list[int]]:
    """Parse bed file names into broad categories and chromatin states.

    Returns:
        Tuple of (broad_categories, chromatin_states, valid_indices).
        Invalid entries are skipped and their indices excluded.
    """
    broad_cats: list[str] = []
    states: list[str] = []
    valid_indices: list[int] = []

    for i, bed in enumerate(bed_names):
        try:
            cell_id, state = classify_bed_file(bed)
            cat = classify_cell_type(cell_id)
            broad = classify_broad_category(cat)
            broad_cats.append(broad)
            states.append(state)
            valid_indices.append(i)
        except ValueError:
            continue

    return broad_cats, states, valid_indices


def get_color_mapping(
    labels: list[str], ordered_categories: list[str], cmap_name: str = "tab20"
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Create color array from labels using category ordering.

    Returns:
        Tuple of (colors array, category_to_color dict).
    """
    cmap = plt.get_cmap(cmap_name)

    # Build color mapping from ordered categories
    category_to_color: dict[str, np.ndarray] = {}
    for i, cat in enumerate(ordered_categories):
        category_to_color[cat] = np.array(cmap(i / max(len(ordered_categories) - 1, 1)))

    # Map labels to colors
    colors = np.array(
        [category_to_color.get(label, [0.5, 0.5, 0.5, 1.0]) for label in labels]
    )
    return colors, category_to_color


def plot_umap_scatter(
    ax: Axes,
    embedding_2d: np.ndarray,
    colors: np.ndarray,
    category_to_color: dict[str, np.ndarray],
    title: str,
    point_size: float = 5,
    alpha: float = 0.7,
) -> None:
    """Plot UMAP scatter with legend."""
    ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=colors,
        s=point_size,
        alpha=alpha,
        edgecolors="none",
    )

    # Create legend handles
    legend_handles = []
    for cat, color in category_to_color.items():
        handle = ax.scatter([], [], c=[color], s=20, label=cat, edgecolors="none")
        legend_handles.append(handle)

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7,
        framealpha=0.8,
        markerscale=1.5,
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def plot_rme_umap(
    embedding_path: Path | None = None,
    output_path: Path | None = None,
    show: bool = True,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    point_size: float = 5,
    alpha: float = 0.7,
    matrix_path: Path | None = None,
    names_file: Path | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Create UMAP plots colored by broad category, cell category, and chromatin state.

    Supports two input modes:
    1. Embeddings mode: Pass embedding_path to a zarr file with bed_names attribute.
    2. Similarity matrix mode: Pass matrix_path and names_file.

    Args:
        embedding_path: Path to zarr embeddings with bed_names attribute.
        output_path: Optional path to save the figure.
        show: Whether to display the plot interactively.
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        random_state: Random seed for reproducibility.
        point_size: Size of scatter points.
        alpha: Transparency of points.
        matrix_path: Path to similarity matrix file (alternative to embedding_path).
        names_file: File with BED names, one per line (required with matrix_path).

    Returns:
        Figure and axes tuple.
    """
    # Determine input mode and load data
    distance_matrix: np.ndarray | None = None
    embeddings: np.ndarray | None = None

    if matrix_path is not None:
        if names_file is None:
            raise ValueError("names_file is required when using a similarity matrix")
        print(f"Loading similarity matrix from {matrix_path}", file=sys.stderr)
        distance_matrix, bed_names = load_similarity_matrix(matrix_path, names_file)
        print(
            f"Loaded {distance_matrix.shape[0]}x{distance_matrix.shape[1]} distance matrix",
            file=sys.stderr,
        )
    elif embedding_path is not None:
        print(f"Loading embeddings from {embedding_path}", file=sys.stderr)
        embeddings, bed_names = load_embeddings(embedding_path)
        print(
            f"Loaded {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}",
            file=sys.stderr,
        )
    else:
        raise ValueError("Either embedding_path or matrix_path must be provided")

    if not bed_names:
        raise ValueError("No bed_names attribute found in zarr file")

    # Parse labels
    broad_cats, states, valid_indices = parse_bed_labels(bed_names)
    print(f"Parsed {len(valid_indices)} valid bed files", file=sys.stderr)

    if len(valid_indices) == 0:
        raise ValueError("No valid bed files found after parsing")

    # Filter to valid entries and compute UMAP
    if distance_matrix is not None:
        # Filter distance matrix to valid entries (rows and columns)
        filtered_distance = distance_matrix[np.ix_(valid_indices, valid_indices)]

        print("Computing UMAP projection from distance matrix...", file=sys.stderr)
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric="precomputed",
            random_state=random_state,
            verbose=False,
        )
        embedding_2d = np.asarray(reducer.fit_transform(filtered_distance))
    elif embeddings is not None:
        # Filter embeddings to valid entries
        filtered_embeddings = embeddings[valid_indices]

        print("Computing UMAP projection...", file=sys.stderr)
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=random_state,
            verbose=False,
        )
        embedding_2d = np.asarray(reducer.fit_transform(filtered_embeddings))
    else:
        # This shouldn't happen due to earlier validation
        raise ValueError("No data loaded")
    print("UMAP projection complete", file=sys.stderr)

    # Get colors for all labelings
    broad_colors, broad_to_color = get_color_mapping(
        broad_cats, broad_categories, "tab10"
    )
    state_colors, state_to_color = get_color_mapping(states, chromatin_states, "tab20")

    # Filter to only categories/states that appear in the data
    broad_to_color = {k: v for k, v in broad_to_color.items() if k in broad_cats}
    state_to_color = {k: v for k, v in state_to_color.items() if k in states}

    # Create figure with constrained layout
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")

    plot_umap_scatter(
        ax1,
        embedding_2d,
        broad_colors,
        broad_to_color,
        "Colored by Broad Category",
        point_size=point_size,
        alpha=alpha,
    )

    plot_umap_scatter(
        ax2,
        embedding_2d,
        state_colors,
        state_to_color,
        "Colored by Chromatin State",
        point_size=point_size,
        alpha=alpha,
    )

    fig.suptitle("Roadmap Epigenomics Embedding UMAP", fontsize=14, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}", file=sys.stderr)

    if show:
        plt.show()

    return fig, (ax1, ax2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot UMAP visualization of Roadmap Epigenomics embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    # From embeddings (zarr)
    uv run src/analysis/rme_umap.py data/rme_embeddings.zarr

    # From similarity matrix
    uv run src/analysis/rme_umap.py --matrix similarity.mat --names bed_names.txt

    # Save to file without displaying
    uv run src/analysis/rme_umap.py data/rme_embeddings.zarr -o umap.png --no-show

    # Adjust UMAP parameters
    uv run src/analysis/rme_umap.py data/rme_embeddings.zarr --n-neighbors 30 --min-dist 0.2
""",
    )
    parser.add_argument(
        "embeddings",
        type=Path,
        nargs="?",
        default=None,
        help="Path to zarr embeddings (with bed_names attribute).",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        help="Path to similarity matrix file (alternative to embeddings).",
    )
    parser.add_argument(
        "--names",
        type=Path,
        help="File with BED names, one per line (required with --matrix).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path (e.g., umap.png, umap.pdf).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot interactively.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter (default: 15).",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=5,
        help="Size of scatter points (default: 5).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Transparency of points (default: 0.7).",
    )

    args = parser.parse_args()

    # Validate input mode
    if args.matrix is not None:
        if args.names is None:
            print("Error: --names is required when using --matrix", file=sys.stderr)
            sys.exit(1)
        if not args.matrix.exists():
            print(f"Error: matrix file not found: {args.matrix}", file=sys.stderr)
            sys.exit(1)
        if not args.names.exists():
            print(f"Error: names file not found: {args.names}", file=sys.stderr)
            sys.exit(1)
    elif args.embeddings is not None:
        if not args.embeddings.exists():
            print(
                f"Error: embeddings file not found: {args.embeddings}", file=sys.stderr
            )
            sys.exit(1)
    else:
        print("Error: either embeddings or --matrix must be provided", file=sys.stderr)
        sys.exit(1)

    plot_rme_umap(
        embedding_path=args.embeddings,
        output_path=args.output,
        show=not args.no_show,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
        point_size=args.point_size,
        alpha=args.alpha,
        matrix_path=args.matrix,
        names_file=args.names,
    )


if __name__ == "__main__":
    main()
