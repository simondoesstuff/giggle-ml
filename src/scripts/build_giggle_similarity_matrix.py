#!/usr/bin/env python3

"""
build_giggle_similarity_matrix.py

Build a similarity matrix from a directory of BED files using giggle.

This script creates a GiggleIndex from a directory of BED files and computes
pairwise similarity scores (combo_score) between all files, storing the result
as a memory-mapped fp16 matrix.
"""

import argparse
import sys
from pathlib import Path

from giggleml.data.giggle import GiggleIndex
from giggleml.data.similarity_matrix import SimilarityMatrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a similarity matrix from BED files using giggle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s /data/beds similarity.mat
  %(prog)s /data/beds similarity.mat --unsorted
  %(prog)s /data/beds similarity.mat --index-dir /tmp/giggle_idx --genome-size 3088269832
""",
    )

    parser.add_argument(
        "bed_dir",
        type=Path,
        help="Directory containing .bed.gz files to index",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output path for the similarity matrix file",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Directory to store the giggle index. Defaults to {bed_dir}.giggle",
    )
    parser.add_argument(
        "--unsorted",
        action="store_true",
        help="Input BED files are NOT sorted (omits giggle index -s flag)",
    )
    parser.add_argument(
        "--genome-size",
        type=int,
        default=None,
        help="Genome size for significance testing (default: giggle default)",
    )
    parser.add_argument(
        "--asymmetric",
        action="store_true",
        help="Don't enforce matrix symmetry (by default, M[a,b] = M[b,a] = max)",
    )

    args = parser.parse_args()

    if not args.bed_dir.is_dir():
        print(f"Error: Directory does not exist: {args.bed_dir}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating giggle index from: {args.bed_dir}")
    index = GiggleIndex(
        args.bed_dir,
        index_dir=args.index_dir,
        sorted=not args.unsorted,
        genome_size=args.genome_size,
    )

    n_beds = len(index.list_beds)
    print(f"Found {n_beds} BED files in index")

    if n_beds == 0:
        print("Error: No BED files found in directory", file=sys.stderr)
        sys.exit(1)

    print(f"Building similarity matrix -> {args.output}")
    matrix = SimilarityMatrix.build_from_giggle(
        index,
        args.output,
        symmetric=not args.asymmetric,
    )

    print(f"Done. Matrix shape: {matrix.n} x {matrix.n}")
    print(f"Matrix file size: {args.output.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
