#!/usr/bin/env python3

"""
build_contrastive_memmap.py

Build a memory-mapped contrastive learning dataset from zarr embeddings and BED files.

This script consolidates embeddings and intervals into two memmap files for fast
preloading during training.

Usage:
    uv run python src/scripts/build_contrastive_memmap.py \\
        data/roadmap_epigenomics/beds \\
        data/roadmap_epigenomics/embeds \\
        --output-dir data/roadmap_epigenomics/contrastive_memmap
"""

import argparse
from pathlib import Path

from giggleml.data.contrastive_memmap import ContrastiveMemmapData


def get_bed_names(bed_dir: Path) -> list[str]:
    """Get sorted list of BED file names from directory."""
    beds = sorted(p.stem for p in bed_dir.glob("*.bed"))
    if not beds:
        raise ValueError(f"No .bed files found in {bed_dir}")
    return beds


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    size: float = size_bytes
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build memmap from zarr embeddings and BED files for fast preloading."
    )
    parser.add_argument(
        "bed_dir",
        type=Path,
        help="Directory containing .bed files",
    )
    parser.add_argument(
        "embedding_dir",
        type=Path,
        help="Directory containing .zarr embedding arrays",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for memmap files",
    )
    args = parser.parse_args()

    bed_dir: Path = args.bed_dir
    embedding_dir: Path = args.embedding_dir
    output_dir: Path = args.output_dir

    print("=== Build Contrastive Memmap ===")
    print(f"BED directory: {bed_dir}")
    print(f"Embedding directory: {embedding_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Get bed names
    bed_names = get_bed_names(bed_dir)
    print(f"Found {len(bed_names)} BED files")
    print()

    # Build memmap
    memmap = ContrastiveMemmapData.build_from_files(
        bed_names=bed_names,
        embedding_dir=embedding_dir,
        bed_dir=bed_dir,
        output_dir=output_dir,
    )

    # Report statistics
    print()
    print("=== Statistics ===")
    print(f"Total intervals: {memmap.metadata.total_intervals:,}")
    print(f"Number of files: {memmap.metadata.num_files}")
    print(f"Embedding dimension: {memmap.metadata.embedding_dim}")

    # File sizes
    data_path = output_dir / "data.mmap"
    metadata_path = output_dir / "metadata.json"

    data_size = data_path.stat().st_size
    meta_size = metadata_path.stat().st_size
    total_size = data_size + meta_size

    print()
    print("=== File Sizes ===")
    print(f"Data:       {format_size(data_size)}")
    print(f"Metadata:   {format_size(meta_size)}")
    print(f"Total:      {format_size(total_size)}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
