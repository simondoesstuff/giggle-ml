#!/usr/bin/env python3

from collections.abc import Sequence
from pathlib import Path

from giggleml.data.fasta import load_fasta
from giggleml.data.intervals import crop_intervals, load_bed, sorted_by_size
from giggleml.inference.torch_inference import embed_intervals
from giggleml.models.hyena_dna import HyenaDNA
from giggleml.utils.file_utils import Pathish, file_stem


def main(
    size: str,
    fasta_path: Pathish,
    bed_paths: Sequence[Pathish],
    out_dir: Pathish,
    vram_cap: float,
):
    model = HyenaDNA(size)
    out_paths = [Path(out_dir, f"{file_stem(bed)}.zarr") for bed in bed_paths]
    intervals = [
        sorted_by_size(
            list(crop_intervals(load_bed(bed), model.seq_max)),
            descending=True,
        )
        for bed in bed_paths
    ]
    embed_intervals(
        model,
        load_fasta(fasta_path),
        intervals,
        out_paths,
        vram_coeffs=model.vram_coeffs,
        vram_cap=vram_cap,
    )


# uv run torchrun --nproc_per_node=4 src/scripts/hyena_dna_many.py
if __name__ == "__main__":
    rme = Path("data/roadmap_epigenomics")
    beds = list((rme / "beds").iterdir())
    main(
        "16k",
        "data/hg/hg38.fa",
        beds,
        rme / "embeds",
        vram_cap=8 * 1024**3,  # 8 GB
    )
