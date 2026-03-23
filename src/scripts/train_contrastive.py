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
from giggleml.evaluation.ndcg import create_ndcg_callback
from giggleml.train.bed_contrastive_learning import (
    ContrastiveTrainingConfig,
    train,
)
from giggleml.train.contrastive_data_loader import BedFileCache
from giggleml.utils.data_split import train_test_val_split

# === Data Paths ===
rme = Path("data/roadmap_epigenomics")
BED_DIR = rme / "beds"
EMBEDDING_DIR = rme / "embeds"
SIMILARITY_MATRIX_PATH = rme / "giggle_similarity.mat"
CHECKPOINT_DIR = Path("data/checkpoints/cmodel_2026-3-22")
MEMMAP_DIR: Path | None = rme / "contrastive_memmap"

# === Training Configuration ===
CONFIG = ContrastiveTrainingConfig(
    # Model architecture
    seq_dim=128,  # HyenaDNA tiny embedding dim
    latent_dim=512,
    num_latents=512,
    shared_per_stack=1,
    num_stacks=4,
    num_heads=8,  # latent_dim / 64
    output_dim=128,
    cross_attn_chunk_size=4096,
    # Training
    peak_learning_rate=2e-3,
    weight_decay=0.01,
    warmup_steps=1500,
    total_steps=50_000,
    temperature=0.15,
    # Similarity binning: evenly spaced bins mapping (0, 50] -> (0, 1]
    # 4 thresholds define 4 edge types (0-3), need 4 corresponding weights
    bin_thresholds=(10, 20, 30, 40),
    bin_weights=(0.25, 0.5, 0.75, 1.0),
    # Batch sampling: batch size is (anchors * (neighbors + 1))
    num_anchors=64,
    neighbors_per_anchor=1,
    max_intervals=30_000,
    # Input dropout (data augmentation): mask random inputs during training
    # - seq only: model learns to rely on intervals
    # - interval only: model learns to rely on seq embeddings
    # - both: position excluded, model learns from fewer intervals
    input_dropout_seq=0.3,
    input_dropout_interval=0.3,
    input_dropout_both=0.05,
    # Data paths (set from constants above)
    embedding_dir=EMBEDDING_DIR,
    bed_dir=BED_DIR,
    memmap_dir=MEMMAP_DIR,
)

# === Random Seed ===
SEED = 42

# === Logging ===
LOG_EVERY = 30
VAL_EVERY = 200
CHECKPOINT_EVERY = 2500
PLOT_LOSS = True  # Show live loss plot in terminal

# === Train/Test/Val Split ===
TEST_FRACTION = 0.1
VAL_FRACTION = 0.1

# === nDCG Evaluation ===
EVAL_EVERY = 1000  # Run nDCG evaluation every N steps
NDCG_K = 10  # Top-K for nDCG metric


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

    # Train/test/val split (test set reserved for separate evaluation script)
    split = train_test_val_split(
        n_beds, test_fraction=TEST_FRACTION, val_fraction=VAL_FRACTION, seed=SEED
    )
    print(f"Split: {split.n_train} train, {split.n_val} val, {split.n_test} test")

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

    # Create shared cache for all BED files (used by training and nDCG eval)
    print("Loading all BED files...")
    cache = BedFileCache(
        bed_names=bed_names,
        embedding_dir=CONFIG.embedding_dir,
        bed_dir=CONFIG.bed_dir,
        memmap_dir=CONFIG.memmap_dir,
        preload=True,
    )

    # Create nDCG evaluation callback using shared cache
    all_bed_data = cache.get_all()
    ndcg_callback = create_ndcg_callback(
        bed_data=all_bed_data,
        similarity_matrix=similarity_matrix,
        bin_thresholds=CONFIG.bin_thresholds,
        bin_weights=CONFIG.bin_weights,
        anchor_indices=None,  # All files as anchors
        batch_size=32,
        k=NDCG_K,
    )
    print(
        f"nDCG@{NDCG_K} evaluation: all {len(all_bed_data)} files as anchors (all-to-all)"
    )

    # Train (uses same cache)
    print("Starting training...")
    train(
        config=CONFIG,
        similarity_matrix=similarity_matrix,
        bed_names=bed_names,
        key=key,
        cache=cache,
        train_indices=split.train,
        val_indices=split.val,
        log_every=LOG_EVERY,
        val_every=VAL_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_dir=CHECKPOINT_DIR,
        plot_loss=PLOT_LOSS,
        eval_callbacks=[(f"nDCG@{NDCG_K}", ndcg_callback)],
        eval_every=EVAL_EVERY,
    )

    print("Training complete!")


if __name__ == "__main__":
    """
    # VRAM Model
    # https://www.desmos.com/calculator/d6rmfta1l0

    export XLA_FLAGS="--xla_gpu_enable_cudnn_fmha=true --xla_gpu_fused_attention_use_cudnn_rng=true"
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
    export PYTHONUNBUFFERED=1
    uv run src/scripts/train_contrastive.py | tee data/checkpoints/cmodel_x/log
    """
    main()
