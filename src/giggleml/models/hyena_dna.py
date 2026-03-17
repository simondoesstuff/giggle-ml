from collections.abc import Iterable
from typing import Any, cast, override

import einx
import torch
from torch import FloatTensor, Tensor
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer

from giggleml.inference.torch_inference import NucleotideModel
from giggleml.utils.vram import VRAMCoeffs

MODEL_CONFIGS: dict[str, tuple[int, str, int, str]] = {
    "1k": (1024, "LongSafari/hyenadna-tiny-1k-seqlen-hf", 128, "e8c1eff"),
    "16k": (16386, "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf", 128, "d79fa37"),
    "32k": (32768, "LongSafari/hyenadna-small-32k-seqlen-hf", 256, "8fe770c"),
    "160k": (160000, "LongSafari/hyenadna-medium-160k-seqlen-hf", 256, "7ebf717"),
    "450k": (450000, "LongSafari/hyenadna-medium-450k-seqlen-hf", 256, "42dedd4"),
    "1m": (1_000_000, "LongSafari/hyenadna-large-1m-seqlen-hf", 256, "0a629ab"),
}

# VRAM coefficients for V(n, k) = a*n*k_fft + b*n + c where k_fft = next_pow2(2k-1)
# Estimated via scripts/estimate_vram_coefficients.py (c set to 0, CI too wide)
_VRAM_COEFFS: dict[str, VRAMCoeffs] = {
    "1k": VRAMCoeffs(a=6546.0, b=348092.0, c=0.0),
    "16k": VRAMCoeffs(a=6966.0, b=337772.0, c=0.0),
    "32k": VRAMCoeffs(a=12393.0, b=749894.0, c=0.0),
    "160k": VRAMCoeffs(a=13178.0, b=673388.0, c=0.0),
    "450k": VRAMCoeffs(a=13723.0, b=483761.0, c=0.0),
    "1m": VRAMCoeffs(a=14904.0, b=251045.0, c=0.0),
}


class HyenaDNA(NucleotideModel[dict[str, Tensor]]):
    """HyenaDNA genomic foundation model for nucleotide sequence embedding.

    Supported sizes: 1k, 16k, 32k, 160k, 450k, 1m
    """

    _model: Any
    _tokenizer: Any

    def __init__(self, size: str = "1k"):
        super().__init__()

        if size not in MODEL_CONFIGS:
            raise ValueError(
                f"Unsupported size {size}. Supported sizes are {list(MODEL_CONFIGS.keys())}"
            )

        max_seq_len, checkpoint, embed_dim, rev = MODEL_CONFIGS[size]

        self._size: str = size
        self.seq_max: int = max_seq_len
        self.edim: int = embed_dim

        self._tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True
        )
        # Use right-padding so padding tokens come after actual content.
        # This ensures embeddings are consistent regardless of padding length,
        # since the causal convolutions only look at preceding context.
        self._tokenizer.padding_side = "right"
        # HyenaDNA cannot be torch.compile()d because Hyena layers use FFT
        # which is based on complex numbers. TorchInductor does not support
        # complex operators.
        self._model = AutoModel.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=rev,
        )

    @override
    def collate(self, sequences: Iterable[str]) -> dict[str, Tensor]:
        """Tokenize nucleotide sequences for embedding.

        Args:
            sequences: Iterable of nucleotide sequences (will be uppercased).

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors.
        """
        seqs = [seq.upper() for seq in sequences]

        for seq in seqs:
            if len(seq) > self.seq_max:
                raise ValueError(
                    f"Sequence length {len(seq)} exceeds limit {self.seq_max}"
                )

        return self._tokenizer(
            seqs,
            padding="longest",
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

    @override
    def forward(self, batch: dict[str, Tensor]) -> FloatTensor:
        """Embed tokenized sequences using masked mean pooling.

        Args:
            batch: Dict from collate() with 'input_ids' and 'attention_mask'.

        Returns:
            Tensor of shape (batch_size, edim) with float16 embeddings.
        """
        device = next(self._model.parameters()).device
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        hidden: Tensor = self._model(input_ids=input_ids).last_hidden_state
        mask = mask.to(hidden.dtype)

        # Masked mean pooling
        mask_expanded = einx.id("b s -> b s 1", mask)
        assert isinstance(mask_expanded, Tensor)
        weighted = hidden * mask_expanded
        lengths = einx.sum("b [s]", mask)
        summed = einx.sum("b [s] d", weighted)
        lengths_expanded = einx.id("b -> b 1", lengths.clamp(min=1e-9))
        assert isinstance(lengths_expanded, Tensor)
        pooled = summed / lengths_expanded

        return cast(FloatTensor, pooled.to(dtype=torch.float16))

    @property
    def vram_coeffs(self) -> VRAMCoeffs:
        """VRAM model coefficients for dynamic batching."""
        return _VRAM_COEFFS[self._size]

    @override
    def __repr__(self) -> str:
        return f"HyenaDNA({self._size})"
