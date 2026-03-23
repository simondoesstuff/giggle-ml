"""nDCG (Normalized Discounted Cumulative Gain) evaluation for embedding models.

Provides metrics for evaluating how well learned embeddings preserve the ranking
implied by ground truth similarity scores (e.g., GIGGLE combo scores).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from numpy.typing import NDArray

from giggleml.data.similarity_matrix import SimilarityMatrix
from giggleml.inference.equinox_inference import embed_dataset
from giggleml.models.cmodel import CModel
from giggleml.train.contrastive_data_loader import BedFileData
from giggleml.utils.equinox import to_f32


def compute_weighted_relevances(
    similarity_matrix: SimilarityMatrix,
    bin_thresholds: tuple[float, ...],
    bin_weights: tuple[float, ...],
    indices: list[int] | None = None,
) -> NDArray[np.float32]:
    """Convert raw GIGGLE similarities to weighted relevance scores.

    Uses the same binning scheme as training to convert raw similarity values
    to relevance weights suitable for nDCG computation.

    Args:
        similarity_matrix: Full pairwise similarity matrix (GIGGLE scores).
        bin_thresholds: Thresholds defining similarity bins. A value v is in
            bin i if bin_thresholds[i] <= v < bin_thresholds[i+1].
        bin_weights: Weight assigned to each bin.
        indices: Optional subset of indices to extract. If None, uses all.

    Returns:
        Weighted relevance matrix of shape (n, n) or (len(indices), len(indices)).
    """
    # Extract subset if indices provided
    if indices is not None:
        arr = similarity_matrix.array[np.ix_(indices, indices)].astype(np.float32)
    else:
        arr = similarity_matrix.array.astype(np.float32)

    # Compute bin assignments
    # Values below first threshold get weight 0
    thresholds = np.array(bin_thresholds, dtype=np.float32)
    weights = np.array(bin_weights, dtype=np.float32)

    # Digitize: bin 0 = below threshold[0], bin i = threshold[i-1] <= x < threshold[i]
    bin_indices = np.digitize(arr, thresholds)

    # Map to weights (bin 0 gets weight 0, bin i gets weights[i-1])
    relevances = np.where(bin_indices > 0, weights[bin_indices - 1], 0.0)

    return relevances.astype(np.float32)


def compute_ndcg_at_k(
    model_similarities: Float[Array, "n"],
    ground_truth_relevances: Float[Array, "n"],
    k: int,
    exclude_idx: int,
) -> float:
    """Compute nDCG@K for a single anchor.

    nDCG measures how well the model's ranking matches the ideal ranking
    based on ground truth relevances.

    Args:
        model_similarities: Cosine similarities from embeddings (anchor vs all).
        ground_truth_relevances: Weighted relevance scores from GIGGLE.
        k: Number of top results to consider.
        exclude_idx: Index of the anchor itself (excluded from ranking).

    Returns:
        nDCG@K score in [0, 1].
    """
    n = model_similarities.shape[0]

    # Mask out self-similarity
    mask = jnp.ones(n, dtype=bool).at[exclude_idx].set(False)
    model_sims = jnp.where(mask, model_similarities, -jnp.inf)
    relevances = ground_truth_relevances

    # Get top-k by model similarity
    top_k_indices = jnp.argsort(model_sims)[::-1][:k]
    top_k_relevances = relevances[top_k_indices]

    # DCG: sum of relevance / log2(rank + 2) for ranks 0..k-1
    # log2(1+1)=1, log2(2+1)=1.58, log2(3+1)=2, ...
    discounts = jnp.log2(jnp.arange(k) + 2)
    dcg = jnp.sum(top_k_relevances / discounts)

    # Ideal DCG: sort by true relevance
    masked_relevances = jnp.where(mask, relevances, -jnp.inf)
    ideal_indices = jnp.argsort(masked_relevances)[::-1][:k]
    ideal_relevances = relevances[ideal_indices]
    idcg = jnp.sum(ideal_relevances / discounts)

    # Handle edge case where IDCG is 0 (no relevant items)
    ndcg = jnp.where(idcg > 0, dcg / idcg, 0.0)

    return float(ndcg)


@dataclass
class NDCGResult:
    """Result of nDCG evaluation over multiple anchors.

    Attributes:
        mean_ndcg: Mean nDCG@K across all anchors.
        per_anchor_ndcg: nDCG@K for each anchor.
        k: The K value used for evaluation.
    """

    mean_ndcg: float
    per_anchor_ndcg: NDArray[np.float32]
    k: int


def evaluate_ndcg(
    embeddings: Float[Array, "n output_dim"],
    ground_truth: NDArray[np.float32],
    anchor_indices: list[int] | None = None,
    k: int = 10,
) -> NDCGResult:
    """Compute mean nDCG@K over anchors.

    Args:
        embeddings: Embeddings for all files, shape (n, output_dim).
        ground_truth: Weighted relevance matrix from compute_weighted_relevances,
            shape (n, n).
        anchor_indices: Indices of anchors to evaluate. If None, evaluates all.
        k: Number of top results to consider.

    Returns:
        NDCGResult with mean and per-anchor nDCG scores.
    """
    n = embeddings.shape[0]

    # Default to all anchors
    if anchor_indices is None:
        anchor_indices = list(range(n))

    # L2 normalize embeddings for cosine similarity
    embeddings = embeddings / (
        jnp.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8
    )

    # Compute pairwise cosine similarities
    similarities: Float[Array, "n n"] = einx.dot(
        "i d, j d -> i j", embeddings, embeddings
    )

    # Convert ground truth to JAX array
    gt_jax = jnp.asarray(ground_truth)

    # Compute nDCG for each anchor
    per_anchor_ndcg = []
    for anchor_idx in anchor_indices:
        ndcg = compute_ndcg_at_k(
            similarities[anchor_idx],
            gt_jax[anchor_idx],
            k=k,
            exclude_idx=anchor_idx,
        )
        per_anchor_ndcg.append(ndcg)

    per_anchor_arr = np.array(per_anchor_ndcg, dtype=np.float32)
    mean_ndcg = float(np.mean(per_anchor_arr))

    return NDCGResult(
        mean_ndcg=mean_ndcg,
        per_anchor_ndcg=per_anchor_arr,
        k=k,
    )


def create_ndcg_callback(
    bed_data: list[BedFileData],
    similarity_matrix: SimilarityMatrix,
    bin_thresholds: tuple[float, ...],
    bin_weights: tuple[float, ...],
    anchor_indices: list[int] | None = None,
    k: int = 10,
    batch_size: int = 64,
) -> Callable[[CModel], float]:
    """Create a callback for train() that computes nDCG@K.

    The callback embeds all bed_data files and computes nDCG@K against the
    ground truth similarity matrix.

    Args:
        bed_data: List of BedFileData objects to evaluate.
        similarity_matrix: Full pairwise similarity matrix.
        bin_thresholds: Thresholds for binning similarity scores.
        bin_weights: Weights for each bin.
        anchor_indices: Indices into bed_data for anchor selection.
            If None, uses all files as anchors.
        k: Number of top results for nDCG@K.
        batch_size: Batch size for embedding computation.

    Returns:
        Callback function that takes a CModel and returns {"nDCG@K": value}.
    """
    # Get file indices from bed_data for extracting ground truth subset
    file_indices = [bd.node_idx for bd in bed_data]

    # Precompute weighted relevances for the evaluation subset
    ground_truth = compute_weighted_relevances(
        similarity_matrix,
        bin_thresholds,
        bin_weights,
        indices=file_indices,
    )

    # Map anchor indices from bed_data indices to evaluation subset indices
    if anchor_indices is not None:
        # anchor_indices are indices into bed_data (0 to len(bed_data)-1)
        eval_anchor_indices = anchor_indices
    else:
        eval_anchor_indices = None

    def callback(model: CModel) -> float:
        """Compute nDCG@K for the given model."""
        jax.clear_caches()
        model = eqx.nn.inference_mode(model)

        # Embed all files
        embeddings = to_f32(embed_dataset(model, bed_data, batch_size=batch_size))

        # Compute nDCG
        result = evaluate_ndcg(
            embeddings,
            ground_truth,
            anchor_indices=eval_anchor_indices,
            k=k,
        )

        return result.mean_ndcg

    return callback
