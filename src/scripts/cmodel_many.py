"""CLI to embed multiple BED files using a CModel checkpoint.

Takes a series of BED files, embeds them using a CModel checkpoint, and writes
the embeddings to a zarr array ordered by sorted bed file names.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import zarr

from giggleml.inference.equinox_inference import embed_dataset
from giggleml.models.cmodel import create_cmodel
from giggleml.train.contrastive_data_loader import BedFileCache
from giggleml.utils.equinox import load_checkpoint, to_bf16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed BED files using a CModel checkpoint and write to zarr."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to CModel checkpoint (.eqx file)",
    )
    parser.add_argument(
        "--bed-dir",
        type=Path,
        required=True,
        help="Directory containing .bed.gz files",
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        required=True,
        help="Directory containing HyenaDNA embeddings (zarr arrays)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output zarr path for embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference (default: 64)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=524_288,
        help="Zarr chunk size along batch dimension (default: 524,288)",
    )
    parser.add_argument(
        "--memmap-dir",
        type=Path,
        default=None,
        help="Optional directory for memory-mapped embeddings",
    )
    # Model architecture params (must match checkpoint)
    parser.add_argument("--seq-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--num-latents", type=int, default=512)
    parser.add_argument("--shared-per-stack", type=int, default=1)
    parser.add_argument("--num-stacks", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--cross-attn-chunk-size", type=int, default=4096)

    return parser.parse_args()


def discover_bed_files(bed_dir: Path) -> list[str]:
    """Discover all .bed and .bed.gz files in directory and return sorted names without suffix."""
    # Find both extensions by combining the glob results
    bed_files = list(bed_dir.glob("*.bed.gz")) + list(bed_dir.glob("*.bed"))
    # Extract names by sequentially stripping possible suffixes
    names = [f.name.removesuffix(".bed.gz").removesuffix(".bed") for f in bed_files]
    # Use set() to ensure we don't return duplicates if both formats exist for the same name, then sort
    return sorted(list(set(names)))


def main() -> None:
    args = parse_args()

    # Discover bed files
    bed_names = discover_bed_files(args.bed_dir)
    if not bed_names:
        raise ValueError(f"No .bed.gz files found in {args.bed_dir}")
    print(f"Found {len(bed_names)} BED files")

    # Create template model for deserialization
    model_template = create_cmodel(
        seq_dim=args.seq_dim,
        latent_dim=args.latent_dim,
        num_latents=args.num_latents,
        shared_per_stack=args.shared_per_stack,
        num_stacks=args.num_stacks,
        num_heads=args.num_heads,
        output_dim=args.output_dim,
        dropout_rate=0,
        pooling="decode",
        cross_attn_chunk_size=args.cross_attn_chunk_size,
        cross_attn_checkpoint=True,
        key=jax.random.key(0),
    )

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    model = to_bf16(load_checkpoint(args.checkpoint, model_template))
    # embed_dataset also sets inference mode, but we can do it here explicitly
    model = eqx.nn.inference_mode(model)
    print("Model loaded")

    # Create cache to load bed file data
    print("Loading BED file data...")
    cache = BedFileCache(
        bed_names=bed_names,
        embedding_dir=args.embedding_dir,
        bed_dir=args.bed_dir,
        memmap_dir=args.memmap_dir,
        preload=True,
    )

    # Get all bed data in sorted order (cache sorts bed_names internally)
    bed_data = cache.get_all()

    # Embed using the high-level inference API
    print(f"Embedding {len(bed_data)} files in batches of {args.batch_size}...")

    embeddings_jax = embed_dataset(
        model=model, bed_data=bed_data, batch_size=args.batch_size, use_tqdm=True
    )

    # Convert jax array back to numpy for Zarr writing
    embeddings_array = np.asarray(embeddings_jax)
    print(f"Final embeddings shape: {embeddings_array.shape}")

    # Write to zarr with large chunk size
    print(f"Writing to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Chunk size: (chunk_size, output_dim) for efficient sequential access
    chunks = (min(args.chunk_size, len(bed_names)), embeddings_array.shape[1])

    z = zarr.open_array(
        str(args.output),
        mode="w",
        shape=embeddings_array.shape,
        chunks=chunks,
        dtype=embeddings_array.dtype,
    )
    z[:] = embeddings_array

    # Also save the bed names as metadata for reference
    z.attrs["bed_names"] = bed_names
    z.attrs["ordered"] = True

    print(f"Done! Wrote {len(bed_names)} embeddings to {args.output}")


# uv run src/scripts/cmodel_many.py --checkpoint data/checkpoints/cmodel_2026-3-23/state_step_20000/model.eqx --bed-dir data/roadmap_epigenomics/beds --embedding-dir data/roadmap_epigenomics/embeds/ --output data/roadmap_epigenomics/cmodel_embeds.zarr --batch-size 16 --memmap-dir data/roadmap_epigenomics/contrastive_memmap/
if __name__ == "__main__":
    main()
