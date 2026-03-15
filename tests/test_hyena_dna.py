"""Tests for giggleml.models.hyena_dna module."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import Tensor

from giggleml.models.hyena_dna import MODEL_CONFIGS, HyenaDNA


class TestModelConfigs:
    """Tests for MODEL_CONFIGS constant."""

    def test_contains_expected_sizes(self):
        expected_sizes = ["1k", "16k", "32k", "160k", "450k", "1m"]
        for size in expected_sizes:
            assert size in MODEL_CONFIGS

    def test_config_structure(self):
        for size, config in MODEL_CONFIGS.items():
            assert len(config) == 4
            seq_max, checkpoint, edim, rev = config
            assert isinstance(seq_max, int)
            assert isinstance(checkpoint, str)
            assert isinstance(edim, int)
            assert isinstance(rev, str)

    def test_1k_config(self):
        seq_max, checkpoint, edim, rev = MODEL_CONFIGS["1k"]
        assert seq_max == 1024
        assert edim == 128
        assert "hyenadna" in checkpoint.lower()

    def test_1m_config(self):
        seq_max, checkpoint, edim, rev = MODEL_CONFIGS["1m"]
        assert seq_max == 1_000_000
        assert edim == 256


class TestHyenaDNAInit:
    """Tests for HyenaDNA initialization."""

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError, match="Unsupported size"):
            with patch(
                "giggleml.models.hyena_dna.AutoTokenizer"
            ), patch("giggleml.models.hyena_dna.AutoModel"):
                HyenaDNA(size="invalid")

    def test_invalid_size_lists_valid_sizes(self):
        with pytest.raises(ValueError) as exc_info:
            with patch(
                "giggleml.models.hyena_dna.AutoTokenizer"
            ), patch("giggleml.models.hyena_dna.AutoModel"):
                HyenaDNA(size="2k")
        assert "1k" in str(exc_info.value)

    @patch("giggleml.models.hyena_dna.AutoModel")
    @patch("giggleml.models.hyena_dna.AutoTokenizer")
    def test_sets_seq_max(self, mock_tokenizer, mock_model):
        model = HyenaDNA(size="1k")
        assert model.seq_max == 1024

    @patch("giggleml.models.hyena_dna.AutoModel")
    @patch("giggleml.models.hyena_dna.AutoTokenizer")
    def test_sets_edim(self, mock_tokenizer, mock_model):
        model = HyenaDNA(size="1k")
        assert model.edim == 128

    @patch("giggleml.models.hyena_dna.AutoModel")
    @patch("giggleml.models.hyena_dna.AutoTokenizer")
    def test_default_size(self, mock_tokenizer, mock_model):
        model = HyenaDNA()
        assert model.seq_max == 1024  # Default is "1k"


class TestHyenaDNACollate:
    """Tests for HyenaDNA.collate method."""

    @patch("giggleml.models.hyena_dna.AutoModel")
    @patch("giggleml.models.hyena_dna.AutoTokenizer")
    def test_uppercases_sequences(self, mock_tokenizer_cls, mock_model):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.zeros(1, 10)}
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        model = HyenaDNA(size="1k")
        model.collate(["acgt", "ACGT"])

        # Check that tokenizer was called with uppercased sequences
        call_args = mock_tokenizer.call_args
        seqs = call_args[0][0]
        assert seqs == ["ACGT", "ACGT"]

    @patch("giggleml.models.hyena_dna.AutoModel")
    @patch("giggleml.models.hyena_dna.AutoTokenizer")
    def test_sequence_too_long_raises(self, mock_tokenizer_cls, mock_model):
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        model = HyenaDNA(size="1k")  # seq_max = 1024

        with pytest.raises(ValueError, match="exceeds limit"):
            model.collate(["A" * 2000])

    @patch("giggleml.models.hyena_dna.AutoModel")
    @patch("giggleml.models.hyena_dna.AutoTokenizer")
    def test_tokenizer_called_with_correct_params(self, mock_tokenizer_cls, mock_model):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.zeros(1, 10)}
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        model = HyenaDNA(size="1k")
        model.collate(["ACGT"])

        mock_tokenizer.assert_called_once()
        call_kwargs = mock_tokenizer.call_args[1]
        assert call_kwargs["max_length"] == 1024
        assert call_kwargs["padding"] == "max_length"
        assert call_kwargs["truncation"] is False
        assert call_kwargs["add_special_tokens"] is False
        assert call_kwargs["return_attention_mask"] is True
        assert call_kwargs["return_tensors"] == "pt"

    @patch("giggleml.models.hyena_dna.AutoModel")
    @patch("giggleml.models.hyena_dna.AutoTokenizer")
    def test_returns_tokenizer_output(self, mock_tokenizer_cls, mock_model):
        expected_output = {
            "input_ids": torch.ones(2, 1024),
            "attention_mask": torch.ones(2, 1024),
        }
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = expected_output
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        model = HyenaDNA(size="1k")
        result = model.collate(["ACGT", "GGGG"])

        assert result is expected_output


class TestHyenaDNAForward:
    """Tests for HyenaDNA.forward method."""

    @patch("giggleml.models.hyena_dna.AutoModel")
    @patch("giggleml.models.hyena_dna.AutoTokenizer")
    def test_returns_correct_shape(self, mock_tokenizer_cls, mock_model_cls):
        # Setup mock model
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.zeros(1)])

        # Mock the hidden state output
        batch_size = 2
        seq_len = 10
        hidden_dim = 128  # edim for 1k model

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(batch_size, seq_len, hidden_dim)
        mock_model.return_value = mock_output
        mock_model_cls.from_pretrained.return_value = mock_model

        model = HyenaDNA(size="1k")
        batch = {
            "input_ids": torch.ones(batch_size, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, seq_len),
        }

        result = model.forward(batch)

        assert result.shape == (batch_size, hidden_dim)

    @patch("giggleml.models.hyena_dna.AutoModel")
    @patch("giggleml.models.hyena_dna.AutoTokenizer")
    def test_returns_float16(self, mock_tokenizer_cls, mock_model_cls):
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.zeros(1)])

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 10, 128)
        mock_model.return_value = mock_output
        mock_model_cls.from_pretrained.return_value = mock_model

        model = HyenaDNA(size="1k")
        batch = {
            "input_ids": torch.ones(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10),
        }

        result = model.forward(batch)

        assert result.dtype == torch.float16

    @patch("giggleml.models.hyena_dna.AutoModel")
    @patch("giggleml.models.hyena_dna.AutoTokenizer")
    def test_masked_mean_pooling(self, mock_tokenizer_cls, mock_model_cls):
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.zeros(1)])

        # Create a simple hidden state for testing pooling
        # Shape: (batch=1, seq=4, hidden=2)
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])

        mock_output = MagicMock()
        mock_output.last_hidden_state = hidden
        mock_model.return_value = mock_output
        mock_model_cls.from_pretrained.return_value = mock_model

        model = HyenaDNA(size="1k")
        model.edim = 2  # Override for test

        # Mask: only first 2 positions are valid
        batch = {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        }

        result = model.forward(batch)

        # Mean of positions 0 and 1: [(1+3)/2, (2+4)/2] = [2.0, 3.0]
        expected = torch.tensor([[2.0, 3.0]], dtype=torch.float16)
        assert torch.allclose(result, expected, atol=1e-2)


class TestHyenaDNARepr:
    """Tests for HyenaDNA.__repr__ method."""

    @patch("giggleml.models.hyena_dna.AutoModel")
    @patch("giggleml.models.hyena_dna.AutoTokenizer")
    def test_repr_format(self, mock_tokenizer, mock_model):
        model = HyenaDNA(size="1k")
        assert repr(model) == "HyenaDNA(1k)"

    @patch("giggleml.models.hyena_dna.AutoModel")
    @patch("giggleml.models.hyena_dna.AutoTokenizer")
    def test_repr_different_sizes(self, mock_tokenizer, mock_model):
        model = HyenaDNA(size="32k")
        assert repr(model) == "HyenaDNA(32k)"


class TestHyenaDNAInheritance:
    """Tests for HyenaDNA inheritance from NucleotideModel."""

    @patch("giggleml.models.hyena_dna.AutoModel")
    @patch("giggleml.models.hyena_dna.AutoTokenizer")
    def test_is_nn_module(self, mock_tokenizer, mock_model):
        model = HyenaDNA(size="1k")
        assert isinstance(model, torch.nn.Module)

    @patch("giggleml.models.hyena_dna.AutoModel")
    @patch("giggleml.models.hyena_dna.AutoTokenizer")
    def test_has_seq_max_attribute(self, mock_tokenizer, mock_model):
        model = HyenaDNA(size="1k")
        assert hasattr(model, "seq_max")

    @patch("giggleml.models.hyena_dna.AutoModel")
    @patch("giggleml.models.hyena_dna.AutoTokenizer")
    def test_has_edim_attribute(self, mock_tokenizer, mock_model):
        model = HyenaDNA(size="1k")
        assert hasattr(model, "edim")


class TestHyenaDNAIntegration:
    """Integration tests that load the actual model.

    These tests require network access to download model weights.
    Mark with @pytest.mark.slow if you want to skip in quick test runs.
    """

    @pytest.fixture(scope="class")
    def model(self):
        """Load model once for all tests in this class."""
        return HyenaDNA(size="1k")

    def test_embedding_output_values(self, model):
        """Test that embeddings for known sequences match expected values."""
        sequences = ["ACGTACGTACGT", "GGGGCCCCAAAA"]

        model.eval()
        with torch.inference_mode():
            batch = model.collate(sequences)
            embeddings = model(batch)

        # Expected values for first 5 dimensions
        # Shape: (2, 128) for 1k model
        expected_first_5_seq1 = torch.tensor(
            [-0.671875, 0.474609375, -0.60546875, 3.0625, 0.4765625], dtype=torch.float16
        )
        expected_first_5_seq2 = torch.tensor(
            [-0.69921875, 0.59765625, -0.5390625, 3.609375, 0.5], dtype=torch.float16
        )

        assert embeddings.shape == (2, 128)
        assert torch.allclose(
            embeddings[0, :5], expected_first_5_seq1, atol=1e-2
        ), f"Seq1 first 5: {embeddings[0, :5].tolist()}"
        assert torch.allclose(
            embeddings[1, :5], expected_first_5_seq2, atol=1e-2
        ), f"Seq2 first 5: {embeddings[1, :5].tolist()}"

    def test_deterministic_output(self, model):
        """Test that same input produces same output."""
        sequence = ["ACGTACGTACGT"]

        model.eval()
        with torch.inference_mode():
            batch = model.collate(sequence)
            emb1 = model(batch)
            emb2 = model(batch)

        assert torch.allclose(emb1, emb2, atol=1e-6)
