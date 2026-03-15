"""
Clean and standardize gene symbols using the mygene API.
Clean list is 1:1 wrt original list.
"""

import argparse
from pathlib import Path

import mygene


def main(input_file: Path, output_file: Path):
    mg = mygene.MyGeneInfo()
    gene_list = input_file.read_text().splitlines()

    # Query for the current symbol ('symbol') and Ensembl ID ('ensembl.gene')
    # scopes="symbol,alias,prev_symbol" checks current, aliases, and retired names
    results = mg.querymany(
        gene_list,
        scopes="symbol,alias,prev_symbol",
        fields="symbol,ensembl.gene",
        species="human",
    )

    clean_map = dict()
    missing = set(gene_list)

    print("Mapping...")

    for res in results:
        original: str = res["query"]

        if "notfound" in res:
            continue

        # Prefer current symbol, fallback to original if unchanged
        clean_map[original] = res.get("symbol", original)
        missing.discard(
            original
        )  # Use discard instead of remove to avoid KeyError on duplicates

    # Create output preserving input order and including missing genes
    cleaned_genes = []
    for gene in gene_list:
        if gene in clean_map:
            cleaned_genes.append(clean_map[gene])
        else:
            cleaned_genes.append(gene)  # Keep original if not found

    # Write results to output file
    output_file.write_text("\n".join(cleaned_genes) + "\n")

    print(f"Mapped {len(clean_map)} genes. {len(missing)} unrecoverable.")
    print(f"Output written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean gene symbols using mygene API",
        epilog="""
Examples:
  python gene_clean.py -i genes.txt -o clean_genes.txt
  uv run python src/scripts/gene_clean.py -i my_gene_list.txt -o output.txt

Input file format:
  One gene symbol per line, e.g.:
    TP53
    BRCA1
    EGFR
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to input file containing gene symbols (one per line)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to output file for cleaned gene symbols",
    )

    args = parser.parse_args()
    main(args.input, args.output)
