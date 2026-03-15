"""Gene renaming utility using HGNC table.

This script renames genes in a source file using an HGNC (HUGO Gene Nomenclature Committee)
table for standardization. It maps previous symbols and aliases to current official symbols.

Usage:
    uv run python src/scripts/gene_clean.py -t hgnc_table.json -s gene_names.txt

Example:
    uv run python src/scripts/gene_clean.py -t $out/hgnc_complete_set.json -s $out/l1000_cp_gene_names.txt

The script performs in-place renaming of the source file.
"""

import argparse
import json
from collections.abc import Callable
from logging import warning
from pathlib import Path


def main(hgnc_table: Path, source: Path):
    def build_table(data):
        symbol_map = dict[str, str]()

        for doc in data["response"]["docs"]:
            curr = doc["symbol"]
            symbol_map[curr] = curr  # Identity map

            # Map previous symbols (e.g., WARS -> WARS1)
            if "prev_symbol" in doc:
                for prev in doc["prev_symbol"]:
                    symbol_map[prev] = curr

            # Map aliases (e.g., specific clone names)
            if "alias_symbol" in doc:
                for alias in doc["alias_symbol"]:
                    symbol_map[alias] = curr

        # manual fixes

        # 1. Renamed to DRC11 (Dynein Regulatory Complex Subunit 11)
        # HGNC updated this recently; IQCA1 is now considered an alias.
        symbol_map["IQCA1"] = "DRC11"
        # 2. Merged into ITFG2 (Integrin Alpha FG-GAP Repeat Containing 2)
        # LOC100507424 was a redundant Entrez entry for the same locus as ITFG2.
        symbol_map["LOC100507424"] = "ITFG2"
        # 3. Renamed to SIMC1P1 (SIMC1 Pseudogene 1)
        # LOC202181 is the Entrez ID for this specific pseudogene.
        symbol_map["LOC202181"] = "SIMC1P1"

        return symbol_map

    with open(hgnc_table, "r") as f:
        table = build_table(json.load(f))

    input_genes = source.read_text().splitlines()  # full in-memory read
    is_valid: Callable[[str], bool] = lambda x: x != "MT-HPR"
    filtered_genes = list(filter(is_valid, input_genes))

    if len(input_genes) != len(filtered_genes):
        warning("Filtering out MT-HPR because it's mitocondrial.")

    rename: Callable[[str], str] = lambda gene: table.get(gene, gene)
    output_genes = list(map(rename, filtered_genes))

    with open(source, "w") as f:
        f.writelines(gene + "\n" for gene in output_genes)

    print(len(output_genes), "genes")
    print(f"> {source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename genes using HGNC table")
    parser.add_argument(
        "-t", "--table", type=Path, required=True, help="HGNC table JSON file"
    )
    parser.add_argument(
        "-s",
        "--source",
        type=Path,
        required=True,
        help="Source gene names file to rename in-place",
    )

    args = parser.parse_args()
    main(args.table, args.source)
