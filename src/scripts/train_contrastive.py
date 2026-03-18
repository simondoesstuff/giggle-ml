#!/usr/bin/env python3

"""
train_contrastive.py

Kick off contrastive learning training for CModel on BED files.

Configuration is hardcoded for reproducibility. Modify the constants below
to adjust training parameters.
"""

from pathlib import Path

import jax

from giggleml.data.similarity_matrix import SimilarityMatrix
from giggleml.train.bed_contrastive_learning import (
    ContrastiveTrainingConfig,
    train,
)

# === Data Paths ===
rme = Path("data/roadmap_epigenomics")
BED_DIR = rme / "beds"
EMBEDDING_DIR = rme / "embeds"
SIMILARITY_MATRIX_PATH = rme / "giggle_similarity.mat"
CHECKPOINT_DIR = Path("data/checkpoints/cmodel_2026-3-18")

# === Training Configuration ===
CONFIG = ContrastiveTrainingConfig(
    # Model architecture
    seq_dim=128,  # HyenaDNA tiny embedding dim
    latent_dim=256,
    num_latents=512,
    shared_per_stack=1,
    num_stacks=4,
    num_heads=8,
    output_dim=128,
    # Training
    peak_learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=1000,
    total_steps=100_000,
    temperature=0.07,
    # Similarity binning: evenly spaced bins mapping (0, 50] -> (0, 1]
    bin_thresholds=(10, 20, 30, 40),
    bin_weights=(0.2, 0.4, 0.6, 0.8, 1.0),
    # Batch sampling
    num_anchors=4,
    neighbors_per_anchor=4,
    # Data paths (set from constants above)
    embedding_dir=EMBEDDING_DIR,
    bed_dir=BED_DIR,
)

# === Random Seed ===
SEED = 42

# === Logging ===
LOG_EVERY = 100
CHECKPOINT_EVERY = None


def get_bed_names(bed_dir: Path) -> list[str]:
    """Get sorted list of BED file names from directory.

    This must match the ordering used when building the similarity matrix.
    """
    beds = sorted(p.stem for p in bed_dir.glob("*.bed"))
    if not beds:
        raise ValueError(f"No .bed files found in {bed_dir}")
    return beds


def main() -> None:
    print("=== Contrastive Learning Training ===")
    print(f"BED directory: {BED_DIR}")
    print(f"Embedding directory: {EMBEDDING_DIR}")
    print(f"Similarity matrix: {SIMILARITY_MATRIX_PATH}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print()

    # Load bed names
    bed_names = get_bed_names(BED_DIR)
    n_beds = len(bed_names)
    print(f"Found {n_beds} BED files")

    # Load similarity matrix
    print(f"Loading similarity matrix ({n_beds} x {n_beds})...")
    similarity_matrix = SimilarityMatrix(
        SIMILARITY_MATRIX_PATH,
        n=n_beds,
        mode="r",
    )

    # Print config summary
    print()
    print("Configuration:")
    print(f"  Model: latent_dim={CONFIG.latent_dim}, num_latents={CONFIG.num_latents}")
    print(f"  Training: lr={CONFIG.peak_learning_rate}, steps={CONFIG.total_steps}")
    print(
        f"  Batch: {CONFIG.num_anchors} anchors x {CONFIG.neighbors_per_anchor} neighbors"
    )
    print(f"  Bins: thresholds={CONFIG.bin_thresholds}, weights={CONFIG.bin_weights}")
    print()

    # Create checkpoint directory
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize PRNG
    key = jax.random.key(SEED)

    # Train
    print("Starting training...")
    train(
        config=CONFIG,
        similarity_matrix=similarity_matrix,
        bed_names=bed_names,
        key=key,
        log_every=LOG_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    print("Training complete!")


if __name__ == "__main__":
    main()
