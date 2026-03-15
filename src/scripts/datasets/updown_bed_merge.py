"""
Script to merge up and down regulated genomic regions from BED files.

For each *_up.bed file found, this script looks for a corresponding *_down.bed file
and merges the intervals from both files. Overlapping intervals are merged into
single intervals. The output is written to a merged .bed file in the same directory.

Example:
    sample_up.bed + sample_down.bed -> sample.bed

The script processes all files in parallel using multiprocessing for efficiency.

Usage:
    python updown_bed_merge.py -i /path/to/bed/files
    python updown_bed_merge.py -i /path/to/bed/files -n 8
    python updown_bed_merge.py  # uses current directory

Arguments:
    -i, --input-dir: Directory containing *_up.bed and *_down.bed files (default: current directory)
    -n, --num-cores: Number of cores for parallel processing (default: all available cores)
"""

import argparse
import glob
import os
from multiprocessing import Pool
from pathlib import Path


def merge_intervals(intervals):
    """Merges overlapping intervals. Expects sorted list of (chrom, start, end)."""
    if not intervals:
        return []

    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        # Check overlap: same chrom and current start <= last end
        if current[0] == last[0] and current[1] <= last[2]:
            # Merge: max of ends
            merged[-1] = (last[0], last[1], max(last[2], current[2]))
        else:
            merged.append(current)
    return merged


def process_sample(up_file):
    try:
        up_path = Path(up_file)
        base_name = up_path.stem[:-3]  # remove '_up' suffix
        down_file = up_path.parent / f"{base_name}_down.bed"
        out_file = up_path.parent / f"{base_name}.bed"

        # Read both files into memory
        intervals = []
        # Combine read logic for speed
        for fpath in (up_file, down_file):
            if os.path.exists(fpath):
                with open(fpath, "r") as f:
                    for line in f:
                        parts = line.split()
                        # specific bed format: chrom, start, end
                        if len(parts) >= 3:
                            intervals.append((parts[0], int(parts[1]), int(parts[2])))

        # Sort: Primary key chrom, secondary key start (integer)
        intervals.sort()

        # Merge
        merged = merge_intervals(intervals)

        # Write output
        with open(out_file, "w") as f:
            for iv in merged:
                f.write(f"{iv[0]}\t{iv[1]}\t{iv[2]}\n")

    except Exception as e:
        print(f"Error processing {base_name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge up and down regulated genomic regions from BED files"
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        default=".",
        help="Input directory containing *_up.bed and *_down.bed files (default: current directory)"
    )
    parser.add_argument(
        "-n", "--num-cores",
        type=int,
        help="Number of cores to use for parallel processing (default: all available cores)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist")
        return
    
    # Find all *_up.bed files in input directory
    pattern = str(input_path / "*_up.bed")
    files = list(glob.glob(pattern))
    
    if not files:
        print(f"No *_up.bed files found in {input_path}")
        return
    
    print(f"Found {len(files)} *_up.bed files to process")
    print(f"Input directory: {input_path}")
    print(f"Output files will be written to the same directory as inputs")
    
    # Use specified number of cores or all available cores
    processes = args.num_cores
    
    with Pool(processes=processes) as pool:
        # imap_unordered is faster as it doesn't preserve order of results
        # chunksize 100 helps reduce IPC overhead for small tasks
        for _ in pool.imap_unordered(process_sample, files, chunksize=100):
            pass
    
    print("Processing complete!")


if __name__ == "__main__":
    main()
