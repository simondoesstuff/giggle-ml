#!/usr/bin/env python3
"""
Convert gene symbols to genomic regions using MyGene.info API.
Requires internet.
Multiple regions for a single gene name is possible. Missing gene names are skipped, but logged.

Usage:
    echo -e "TP53\nBRCA1\nEGFR" | python scripts/datasets/genes2regions.py -o output.bed

    OR

    cat genes.txt | python scripts/datasets/genes2regions.py -o output.bed

Output BED format: chr, start, end, gene_name (tab-separated).
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import mygene


def read_gene_symbols_from_stdin() -> list[str]:
    """Read gene symbols from stdin, one per line."""
    gene_symbols = []
    for line in sys.stdin:
        line = line.strip()
        if line:
            gene_symbols.append(line)
    return gene_symbols


def query_gene_coordinates(gene_symbols: list[str]) -> list[dict[str, Any]]:
    """Query MyGene.info API for genomic coordinates."""
    mg = mygene.MyGeneInfo()
    
    # Query for genomic coordinates
    results = mg.querymany(
        gene_symbols,
        scopes="symbol,alias,prev_symbol",
        fields="genomic_pos,symbol",
        species="human",
    )
    
    return results


def extract_bed_regions(results: list[dict[str, Any]]) -> tuple[list[tuple[str, int, int, str]], list[str], list[str]]:
    """Extract BED regions from MyGene results."""
    bed_regions = []
    missing_genes = []
    genes_without_coords = []
    
    for res in results:
        gene_name = res["query"]
        
        if "notfound" in res:
            missing_genes.append(gene_name)
            continue
            
        if "genomic_pos" not in res:
            genes_without_coords.append(gene_name)
            continue
            
        # Get the official gene symbol for output
        official_symbol = res.get("symbol", gene_name)
        
        # Handle genomic_pos which can be a list or single entry
        genomic_pos = res["genomic_pos"]
        if not isinstance(genomic_pos, list):
            genomic_pos = [genomic_pos]
            
        for pos in genomic_pos:
            if all(key in pos for key in ["chr", "start", "end"]):
                # Format chromosome (remove 'chr' prefix if present for consistency)
                chr_name = pos["chr"]
                if not chr_name.startswith("chr"):
                    chr_name = f"chr{chr_name}"
                    
                bed_regions.append((chr_name, pos["start"], pos["end"], official_symbol))
            else:
                genes_without_coords.append(gene_name)
                
    return bed_regions, missing_genes, genes_without_coords


def write_bed_file(bed_regions: list[tuple[str, int, int, str]], output_path: Path) -> None:
    """Write BED regions to output file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for chr_name, start, end, gene_name in bed_regions:
            f.write(f"{chr_name}\t{start}\t{end}\t{gene_name}\n")


def main(output_file: Path) -> None:
    # Read gene symbols from stdin
    gene_symbols = read_gene_symbols_from_stdin()
    
    if not gene_symbols:
        print("Error: No gene symbols provided via stdin", file=sys.stderr)
        sys.exit(1)
        
    print(f"Processing {len(gene_symbols)} gene symbols...")
    
    # Query MyGene API
    results = query_gene_coordinates(gene_symbols)
    
    # Extract BED regions
    bed_regions, missing_genes, genes_without_coords = extract_bed_regions(results)
    
    # Write output
    if bed_regions:
        write_bed_file(bed_regions, output_file)
        print(f"Wrote {len(bed_regions)} regions to {output_file}")
    else:
        print("No valid regions found to write", file=sys.stderr)
        
    # Report issues
    if missing_genes:
        print(f"\nWarning: {len(missing_genes)} genes not found:", file=sys.stderr)
        for gene in sorted(missing_genes):
            print(f"  {gene}", file=sys.stderr)
            
    if genes_without_coords:
        print(f"\nWarning: {len(genes_without_coords)} genes had no genomic coordinates:", file=sys.stderr)
        for gene in sorted(genes_without_coords):
            print(f"  {gene}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert gene symbols to genomic regions using MyGene.info API"
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output BED file path"
    )
    
    args = parser.parse_args()
    main(args.output)
