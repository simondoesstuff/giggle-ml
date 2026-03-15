#!/usr/bin/env python3

from collections.abc import Sequence
from pathlib import Path

from giggleml.data.fasta import load_fasta
from giggleml.data.intervals import crop_intervals, load_bed
from giggleml.inference.torch_inference import embed_intervals
from giggleml.models.hyena_dna import HyenaDNA
from giggleml.utils.file_utils import Pathish, file_stem


def main(
    size: str,
    fasta_path: Pathish,
    bed_paths: Sequence[Pathish],
    out_dir: Pathish,
    batch_size: int,
):
    model = HyenaDNA(size)
    out_paths = [Path(out_dir, file_stem(bed)) for bed in bed_paths]
    intervals = [crop_intervals(load_bed(bed), model.seq_max) for bed in bed_paths]
    embed_intervals(
        model, load_fasta(fasta_path), intervals, out_paths, batch_size=batch_size
    )


if __name__ == "__main__":
    batch_size = 76
    rme = Path("data/roadmap_epigenomics")
    beds = list((rme / "beds").iterdir())[950:]
    main("16k", "data/hg/hg38.fa", beds, rme / "embeds", batch_size)
