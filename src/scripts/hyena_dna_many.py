#!/usr/bin/env python3

"""CLI to embed multiple BED files using a HyenaDNA model.

Takes a directory of BED files, embeds them using a specified HyenaDNA model,
and writes the embeddings to zarr arrays. Designed to be run with torchrun
for multi-GPU distribution.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from giggleml.data.fasta import load_fasta
from giggleml.data.intervals import (
    crop_intervals,
    filter_chromosomes,
    load_bed,
    sorted_by_size,
)
from giggleml.inference.torch_inference import embed_intervals
from giggleml.models.hyena_dna import HyenaDNA
from giggleml.utils.file_utils import Pathish, file_stem, possibly_gzipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed BED files using a HyenaDNA model."
    )
    parser.add_argument(
        "--size",
        type=str,
        default="16k",
        choices=["1k", "16k", "32k", "160k", "450k", "1m"],
        help="HyenaDNA model size (default: 16k)",
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        required=True,
        help="Path to the reference FASTA file (.fa or .fasta)",
    )
    bed_input = parser.add_mutually_exclusive_group(required=True)
    bed_input.add_argument(
        "--bed-dir",
        type=Path,
        help="Directory containing .bed or .bed.gz files",
    )
    bed_input.add_argument(
        "--bed-file",
        type=Path,
        help="Single .bed or .bed.gz file to embed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the resulting zarr arrays",
    )
    parser.add_argument(
        "--vram-cap",
        type=float,
        default=10.0,
        help="VRAM cap per GPU in GiB (default: 10.0)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250000,
        help="Number of embeddings per Zarr chunk (default: 250000)",
    )
    return parser.parse_args()


def main(
    size: str,
    fasta_path: Pathish,
    bed_paths: Sequence[Pathish],
    out_dir: Pathish,
    vram_cap: float,
    zarr_chunk_size: int,
):
    print(f"Loading HyenaDNA model ({size})...")
    model = HyenaDNA(size)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_paths = [out_dir_path / f"{file_stem(bed)}.zarr" for bed in bed_paths]

    print(f"Loading and cropping intervals from {len(bed_paths)} BED files...")
    intervals = [
        sorted_by_size(
            list(crop_intervals(filter_chromosomes(load_bed(bed)), model.seq_max)),
            descending=True,
        )
        for bed in bed_paths
    ]

    print(f"Embedding intervals (VRAM cap: {vram_cap / 1024**3:.1f} GiB)...")
    embed_intervals(
        model,
        load_fasta(fasta_path),
        intervals,
        out_paths,
        vram_coeffs=model.vram_coeffs,
        vram_cap=vram_cap,
        zarr_chunk_size=zarr_chunk_size,
    )


# Example execution:
# uv run torchrun --nproc_per_node=4 src/scripts/hyena_dna_many.py --bed-dir data/beds --output-dir data/embeds --fasta data/hg/hg38.fa
# uv run torchrun --nproc_per_node=4 src/scripts/hyena_dna_many.py --bed-file data/beds/sample.bed --output-dir data/embeds --fasta data/hg/hg38.fa
if __name__ == "__main__":
    args = parse_args()

    if args.bed_file is not None:
        # Single file mode - resolve .gz variant if needed
        bed_paths = [possibly_gzipped(args.bed_file)]
    else:
        # Directory mode - discover both .bed and .bed.gz files
        bed_files = list(args.bed_dir.glob("*.bed.gz")) + list(
            args.bed_dir.glob("*.bed")
        )

        # Deduplicate in case there are identical base names
        bed_paths = list(set(bed_files))
        bed_paths.sort()

        if not bed_paths:
            raise ValueError(f"No .bed or .bed.gz files found in {args.bed_dir}")

    # Convert GiB to bytes for the inference function
    vram_cap_bytes = args.vram_cap * (1024**3)

    main(
        size=args.size,
        fasta_path=args.fasta,
        bed_paths=bed_paths,
        out_dir=args.output_dir,
        vram_cap=vram_cap_bytes,
        zarr_chunk_size=args.chunk_size,
    )
