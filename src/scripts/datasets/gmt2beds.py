#!/usr/bin/env python3
"""
Convert GMT files to multiple BED files using gene name mappings.

Usage:
    cat file.gmt | python gmt2beds.py -f from_genes.txt -t to_genes.txt -r regions.bed -o output_dir

Input:
    - GMT files piped to stdin with format: name\tdescription\tgene1\tgene2\tgene3
    - Gene mapping files (-f from, -t to): newline-delimited gene name lists
    - BED file (-r): contains genomic regions with gene names in extra columns
    - Output directory (-o): where to write {name}_{description}.bed files

Workflow:
1. Parse gene mapping files to create from->to mapping
2. Parse BED file to create gene->regions mapping
3. For each GMT gene set, map gene names and output corresponding regions
4. Report genes that couldn't be mapped at the end
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from giggleml.data_wrangling.interval_dataset import BedDataset


def load_gene_mapping(from_file: Path, to_file: Path) -> dict[str, str]:
    """Load gene name mapping from 'from' genes to 'to' genes."""
    with open(from_file) as f:
        from_genes = [line.strip() for line in f if line.strip()]

    with open(to_file) as f:
        to_genes = [line.strip() for line in f if line.strip()]

    if len(from_genes) != len(to_genes):
        raise ValueError(
            f"Gene mapping files must have same length: {len(from_genes)} vs {len(to_genes)}"
        )

    return dict(zip(from_genes, to_genes))


def load_gene_regions(bed_file: Path) -> dict[str, list[tuple[str, int, int]]]:
    """Load BED file and create mapping from gene names to genomic regions."""
    bed_dataset = BedDataset(bed_file)
    gene_regions = defaultdict(list)

    for interval, extra_cols in bed_dataset.iter_with_extra_columns():
        if extra_cols:
            gene_name = extra_cols[0]  # Gene name is in first extra column
            gene_regions[gene_name].append(interval)

    return dict(gene_regions)


def parse_gmt_stdin() -> list[tuple[str, str, list[str]]]:
    """Parse GMT format from stdin. Returns (name, description, genes) tuples."""
    gene_sets = []

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        name = parts[0]
        description = parts[1]
        genes = parts[2:]

        gene_sets.append((name, description, genes))

    return gene_sets


def write_bed_file(output_path: Path, regions: list[tuple[str, int, int]]):
    """Write regions to BED file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for chrom, start, end in regions:
            f.write(f"{chrom}\t{start}\t{end}\n")


def main(from_file: Path, to_file: Path, regions_file: Path, output_dir: Path):
    # Load mappings
    print("loading gene name mappings...")
    gene_mapping = load_gene_mapping(from_file, to_file)
    print("loading regions...")
    gene_regions = load_gene_regions(regions_file)

    # Parse GMT from stdin
    gene_sets = parse_gmt_stdin()

    # Track unmappable genes for reporting
    unmapped_genes = set()
    genes_without_regions = set()

    # Process each gene set
    for name, description, genes in gene_sets:
        all_regions = []

        for gene in genes:
            # Map gene name
            mapped_gene = gene_mapping.get(gene)
            if mapped_gene is None:
                unmapped_genes.add(gene)
                continue

            # Get regions for mapped gene
            regions = gene_regions.get(mapped_gene, [])
            if not regions:
                genes_without_regions.add(mapped_gene)
                continue

            all_regions.extend(regions)

        # Write BED file if we have regions
        if all_regions:
            output_filename = f"{name}_{description}.bed"
            output_path = output_dir / output_filename
            write_bed_file(output_path, all_regions)
            print(f"Wrote {len(all_regions)} regions to {output_path}")

    # Report issues
    if unmapped_genes:
        print(
            f"\nWarning: {len(unmapped_genes)} genes could not be mapped:",
            file=sys.stderr,
        )
        for gene in sorted(unmapped_genes):
            print(f"  {gene}", file=sys.stderr)

    if genes_without_regions:
        print(
            f"\nWarning: {len(genes_without_regions)} mapped genes had no regions:",
            file=sys.stderr,
        )
        for gene in sorted(genes_without_regions):
            print(f"  {gene}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert GMT files to BED files using gene mappings"
    )
    parser.add_argument(
        "-f",
        "--from",
        required=True,
        type=Path,
        dest="from_file",
        help='File with "from" gene names (newline delimited)',
    )
    parser.add_argument(
        "-t",
        "--to",
        required=True,
        type=Path,
        dest="to_file",
        help='File with "to" gene names (newline delimited)',
    )
    parser.add_argument(
        "-r",
        "--regions",
        required=True,
        type=Path,
        help="BED file with regions and gene names in extra column",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output directory for BED files",
    )

    args = parser.parse_args()
    main(args.from_file, args.to_file, args.regions, args.output)
