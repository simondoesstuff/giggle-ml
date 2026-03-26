"""CLI to compute cosine similarity from CModel embeddings.

Takes query embeddings and reference embeddings (zarr arrays from cmodel_many.py),
computes cosine similarity, and outputs results as TSV (single query) or zarr (matrix).

Supports three modes:
  1. Single query by index: Select one embedding from a multi-entry zarr using --query-index,
     compare against the same file (self) or a --reference set. Outputs TSV.
  2. Single-entry query zarr: If --query contains exactly one embedding and --reference is
     provided, automatically treats it as a one-vs-many comparison. Outputs TSV.
  3. Full matrix: Compute all pairwise similarities between query and reference sets
     (or self-similarity if no --reference). Outputs zarr.

TSV output is sorted by descending similarity with columns: bed_file, similarity_score.
Zarr output stores the full matrix with query_names and reference_names as attrs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import einx
import jax.numpy as jnp
import numpy as np
import zarr
from jaxtyping import Array, Float

EXAMPLES = """\
Examples:
  # 1. Single query by index → TSV (one embedding from a set vs the rest)
  uv run src/scripts/build_embedding_similarity_matrix.py \\
    --query data/cmodel_embeds.zarr \\
    --query-index 0 \\
    --output results/query0_similarities.tsv

  # 2. Single query by index vs separate reference set → TSV
  uv run src/scripts/build_embedding_similarity_matrix.py \\
    --query data/cmodel_embeds.zarr \\
    --query-index 5 \\
    --reference data/other_embeds.zarr \\
    --output results/query5_vs_other.tsv

  # 3. Single-entry query zarr vs reference set → TSV (auto-detected)
  uv run src/scripts/build_embedding_similarity_matrix.py \\
    --query my_single_embedding.zarr \\
    --reference data/cmodel_embeds.zarr \\
    --output results/my_query_similarities.tsv

  # 4. Full self-similarity matrix → zarr
  uv run src/scripts/build_embedding_similarity_matrix.py \\
    --query data/cmodel_embeds.zarr \\
    --output data/self_similarity.zarr

  # 5. Full cross-similarity matrix (query set vs reference set) → zarr
  uv run src/scripts/build_embedding_similarity_matrix.py \\
    --query data/query_embeds.zarr \\
    --reference data/reference_embeds.zarr \\
    --output data/cross_similarity.zarr
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cosine similarity from CModel embeddings.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=Path,
        required=True,
        help="Path to query embeddings zarr (from cmodel_many.py). "
        "If this contains a single embedding and --reference is provided, "
        "outputs TSV automatically.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Path to reference embeddings zarr. If omitted, computes "
        "self-similarity using --query as both query and reference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path. Use .tsv for single-query mode (sorted by similarity), "
        ".zarr for full matrix output.",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        default=None,
        help="Select a single embedding by index from --query for one-vs-many "
        "comparison. Outputs TSV. Useful when query zarr contains multiple "
        "embeddings but you want to compare just one.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Zarr chunk size for matrix output (default: 1024).",
    )
    return parser.parse_args()


def load_embeddings(path: Path) -> tuple[Float[Array, "n d"], list[str]]:
    """Load embeddings from zarr and convert to normalized JAX array.

    Returns:
        Tuple of (L2-normalized embeddings as JAX array, list of bed names).
    """
    z = zarr.open_array(str(path), mode="r")
    embeddings = jnp.asarray(z[:], dtype=jnp.float32)

    # L2 normalize for cosine similarity
    norms = jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    raw_names = z.attrs.get("bed_names", [])
    bed_names: list[str] = (
        [str(n) for n in raw_names] if isinstance(raw_names, list) else []
    )
    return embeddings, bed_names


def compute_cosine_similarity(
    query: Float[Array, "n d"], reference: Float[Array, "m d"]
) -> Float[Array, "n m"]:
    """Compute cosine similarity matrix between query and reference embeddings.

    Both inputs should already be L2-normalized.
    """
    return einx.dot("n d, m d -> n m", query, reference)


def write_tsv(
    similarities: Float[Array, "m"],
    bed_names: list[str],
    output: Path,
) -> None:
    """Write similarity scores to TSV file."""
    scores = np.asarray(similarities)

    # Sort by similarity (descending)
    sorted_indices = np.argsort(scores)[::-1]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["#bed_file", "similarity_score"])
        for idx in sorted_indices:
            writer.writerow([bed_names[idx], f"{scores[idx]:.6f}"])

    print(f"Wrote {len(bed_names)} similarities to {output}")


def write_zarr_matrix(
    similarity_matrix: Float[Array, "n m"],
    query_names: list[str],
    ref_names: list[str],
    output: Path,
    chunk_size: int,
    self_similarity: bool,
) -> None:
    """Write full similarity matrix to zarr."""
    similarity_np = np.asarray(similarity_matrix, dtype=np.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    chunks = (
        min(chunk_size, similarity_np.shape[0]),
        min(chunk_size, similarity_np.shape[1]),
    )

    z = zarr.open_array(
        str(output),
        mode="w",
        shape=similarity_np.shape,
        chunks=chunks,
        dtype=np.float32,
    )
    z[:] = similarity_np

    z.attrs["query_names"] = query_names
    z.attrs["reference_names"] = ref_names
    z.attrs["self_similarity"] = self_similarity

    print(f"Wrote {similarity_np.shape[0]}x{similarity_np.shape[1]} matrix to {output}")


def main() -> None:
    args = parse_args()

    # Load query embeddings
    print(f"Loading query embeddings from {args.query}")
    query_embeddings, query_names = load_embeddings(args.query)
    print(
        f"Query: {query_embeddings.shape[0]} embeddings, dim={query_embeddings.shape[1]}"
    )

    # Load reference embeddings (or use query for self-similarity)
    self_similarity = args.reference is None
    if not self_similarity:
        print(f"Loading reference embeddings from {args.reference}")
        ref_embeddings, ref_names = load_embeddings(args.reference)
        print(
            f"Reference: {ref_embeddings.shape[0]} embeddings, dim={ref_embeddings.shape[1]}"
        )
    else:
        print("Using query as reference (self-similarity)")
        ref_embeddings = query_embeddings
        ref_names = query_names

    # Determine if this is a single-query case
    is_single_query = False
    query_vec = None

    if args.query_index is not None:
        # Explicit index into query set
        if args.query_index < 0 or args.query_index >= query_embeddings.shape[0]:
            print(
                f"Error: query-index {args.query_index} out of range [0, {query_embeddings.shape[0]})",
                file=sys.stderr,
            )
            sys.exit(1)
        query_name = (
            query_names[args.query_index] if query_names else str(args.query_index)
        )
        query_vec = query_embeddings[args.query_index : args.query_index + 1]
        is_single_query = True
        print(
            f"Computing similarity for query '{query_name}' (index {args.query_index})..."
        )

    elif query_embeddings.shape[0] == 1 and not self_similarity:
        # Single-entry query zarr against separate reference set
        query_name = query_names[0] if query_names else "query"
        query_vec = query_embeddings
        is_single_query = True
        print(
            f"Computing similarity for single query '{query_name}' against reference set..."
        )

    # Compute similarity
    if is_single_query:
        assert query_vec is not None
        similarities = compute_cosine_similarity(query_vec, ref_embeddings)[0]
        write_tsv(similarities, ref_names, args.output)
    else:
        # Full matrix
        print("Computing full similarity matrix...")
        similarity_matrix = compute_cosine_similarity(query_embeddings, ref_embeddings)
        print(f"Similarity matrix shape: {similarity_matrix.shape}")

        if args.output.suffix == ".tsv":
            print(
                "Error: TSV output requires --query-index or single-entry query zarr",
                file=sys.stderr,
            )
            sys.exit(1)

        write_zarr_matrix(
            similarity_matrix,
            query_names,
            ref_names,
            args.output,
            args.chunk_size,
            self_similarity,
        )

    print("Done!")


if __name__ == "__main__":
    main()
