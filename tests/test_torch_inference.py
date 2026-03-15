"""Tests for giggleml.inference.torch_inference module."""

from collections.abc import Iterable
from pathlib import Path
from typing import override

import torch
import zarr
from torch import FloatTensor, Tensor

from giggleml.inference.torch_inference import NucleotideModel, embed_intervals


class MockNucleotideModel(NucleotideModel[Tensor]):
    """A simple mock model for testing."""

    def __init__(self, seq_max: int = 100, edim: int = 8):
        super().__init__()
        self.seq_max = seq_max
        self.edim = edim
        self._linear = torch.nn.Linear(4, edim)

    @override
    def collate(self, sequences: Iterable[str]) -> Tensor:
        """Simple tokenization: encode each nucleotide as one-hot."""
        seqs = list(sequences)
        batch_size = len(seqs)
        # Simple encoding: just use sequence length as a feature
        return torch.tensor([[len(s)] * 4 for s in seqs], dtype=torch.float32)

    @override
    def forward(self, batch: Tensor) -> FloatTensor:
        """Simple forward pass."""
        return self._linear(batch)


class TestNucleotideModel:
    """Tests for NucleotideModel abstract class."""

    def test_subclass_has_seq_max(self):
        model = MockNucleotideModel(seq_max=512)
        assert model.seq_max == 512

    def test_subclass_has_edim(self):
        model = MockNucleotideModel(edim=256)
        assert model.edim == 256

    def test_collate_returns_batch(self):
        model = MockNucleotideModel()
        batch = model.collate(["ACGT", "GGGG"])
        assert isinstance(batch, Tensor)
        assert batch.shape[0] == 2

    def test_forward_returns_embeddings(self):
        model = MockNucleotideModel(edim=16)
        batch = model.collate(["ACGT", "GGGG"])
        embeddings = model(batch)
        assert embeddings.shape == (2, 16)

    def test_forward_dtype(self):
        model = MockNucleotideModel()
        batch = model.collate(["ACGT"])
        embeddings = model(batch)
        assert isinstance(embeddings, Tensor)


class TestEmbedIntervals:
    """Tests for embed_intervals function.

    These tests run in single-process mode without CUDA.
    """

    def test_creates_zarr_output(self, tmp_path: Path):
        model = MockNucleotideModel(edim=8)
        fasta = {"chr1": "ACGTACGTACGT", "chr2": "GGGGCCCCAAAA"}
        intervals = [[("chr1", 0, 4), ("chr1", 4, 8)]]
        out_paths = [tmp_path / "out1.zarr"]

        embed_intervals(model, fasta, intervals, out_paths, batch_size=2)

        assert out_paths[0].exists()

    def test_zarr_shape(self, tmp_path: Path):
        model = MockNucleotideModel(edim=8)
        fasta = {"chr1": "ACGTACGTACGT"}
        intervals = [[("chr1", 0, 4), ("chr1", 4, 8), ("chr1", 0, 4)]]
        out_paths = [tmp_path / "out.zarr"]

        embed_intervals(model, fasta, intervals, out_paths, batch_size=2)

        arr = zarr.open(out_paths[0], mode="r")
        assert arr.shape == (3, 8)

    def test_zarr_dtype(self, tmp_path: Path):
        model = MockNucleotideModel(edim=8)
        fasta = {"chr1": "ACGTACGTACGT"}
        intervals = [[("chr1", 0, 4)]]
        out_paths = [tmp_path / "out.zarr"]

        embed_intervals(model, fasta, intervals, out_paths, batch_size=2)

        arr = zarr.open(out_paths[0], mode="r")
        assert arr.dtype.name == "float32"

    def test_multiple_interval_sets(self, tmp_path: Path):
        model = MockNucleotideModel(edim=8)
        fasta = {"chr1": "ACGTACGTACGT"}
        intervals = [
            [("chr1", 0, 4), ("chr1", 4, 8)],
            [("chr1", 0, 4)],
        ]
        out_paths = [tmp_path / "out1.zarr", tmp_path / "out2.zarr"]

        embed_intervals(model, fasta, intervals, out_paths, batch_size=2)

        arr1 = zarr.open(out_paths[0], mode="r")
        arr2 = zarr.open(out_paths[1], mode="r")
        assert arr1.shape == (2, 8)
        assert arr2.shape == (1, 8)

    def test_batching(self, tmp_path: Path):
        model = MockNucleotideModel(edim=8)
        fasta = {"chr1": "A" * 100}
        # 5 intervals with batch_size=2 should process in 3 batches
        intervals = [[("chr1", 0, 4)] * 5]
        out_paths = [tmp_path / "out.zarr"]

        embed_intervals(model, fasta, intervals, out_paths, batch_size=2)

        arr = zarr.open(out_paths[0], mode="r")
        assert arr.shape == (5, 8)

    def test_cleans_up_lock_dir(self, tmp_path: Path):
        model = MockNucleotideModel(edim=8)
        fasta = {"chr1": "ACGTACGTACGT"}
        intervals = [[("chr1", 0, 4)]]
        out_paths = [tmp_path / "out.zarr"]

        embed_intervals(model, fasta, intervals, out_paths, batch_size=2)

        lock_dir = tmp_path / ".embed_locks"
        assert not lock_dir.exists()

    def test_creates_parent_directories(self, tmp_path: Path):
        model = MockNucleotideModel(edim=8)
        fasta = {"chr1": "ACGTACGTACGT"}
        intervals = [[("chr1", 0, 4)]]
        out_paths = [tmp_path / "subdir" / "nested" / "out.zarr"]

        embed_intervals(model, fasta, intervals, out_paths, batch_size=2)

        assert out_paths[0].exists()

    def test_model_in_eval_mode(self, tmp_path: Path):
        model = MockNucleotideModel(edim=8)
        fasta = {"chr1": "ACGTACGTACGT"}
        intervals = [[("chr1", 0, 4)]]
        out_paths = [tmp_path / "out.zarr"]

        # Model should be put in eval mode
        assert model.training  # Initially in training mode

        embed_intervals(model, fasta, intervals, out_paths, batch_size=2)

        assert not model.training  # Should be in eval mode after

    def test_empty_interval_set(self, tmp_path: Path):
        model = MockNucleotideModel(edim=8)
        fasta = {"chr1": "ACGTACGTACGT"}
        intervals = [[]]  # Empty interval set
        out_paths = [tmp_path / "out.zarr"]

        embed_intervals(model, fasta, intervals, out_paths, batch_size=2)

        arr = zarr.open(out_paths[0], mode="r")
        assert arr.shape == (0, 8)
