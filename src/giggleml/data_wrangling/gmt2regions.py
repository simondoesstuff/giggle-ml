from functools import cached_property
from pathlib import Path

import torch
from torch import Tensor

from giggleml.data_wrangling.interval_dataset import (
    IntervalDataset,
)
from giggleml.utils.interval_arithmetic import intervals_to_tensor
from giggleml.utils.symbol_table import SymbolTable
from giggleml.utils.types import lazy


@lazy
class GeneMap:
    @staticmethod
    def build(map_path: IntervalDataset) -> "GeneMap":
        gene_names = []
        for _, extra_cols in map_path.iter_with_extra_columns():
            if extra_cols:
                gene_names.append(extra_cols[0])
            else:
                raise ValueError("Expected gene name in first extra column")

        symbol_table = SymbolTable(list(set(gene_names)))
        return GeneMap(map_path, symbol_table)

    # first extra column will hold the gene name.
    # multiple rows may have the same gene name.
    def __init__(self, map_path: IntervalDataset, symbol_table: SymbolTable[str]):
        self.map_path: IntervalDataset = map_path
        self.symbol_table: SymbolTable[str] = symbol_table

    @cached_property
    def regions(self) -> list[Tensor]:
        # we expect each row to have a single extra column corresponding to gene name.
        # note intervals_to_tensor(...) utility
        gene_intervals = {gene: [] for gene in self.symbol_table.base}

        for interval, extra_cols in self.map_path.iter_with_extra_columns():
            if extra_cols:
                gene_name = extra_cols[0]
                gene_intervals[gene_name].append(interval)

        result = []
        for gene in self.symbol_table.base:
            intervals_for_gene = gene_intervals[gene]
            if intervals_for_gene:
                result.append(intervals_to_tensor(intervals_for_gene, pin_memory=False))
            else:
                # Create empty tensor with correct shape for genes with no intervals
                result.append(torch.empty(0, 3, dtype=torch.int32))
        return result


@lazy
class UpDownGMT2Regions:
    # example rows:
    #   name up gene1 gene2 gene3
    #   name down gene1 gene2 gene3
    #   name2 up gene1 gene2 gene3
    #   name2 down gene1 gene2 gene3

    def __init__(
        self,
        gmt_path: Path,
        gene_map: GeneMap,
    ):
        self.gmt_path: Path = gmt_path
        self.gene_map: GeneMap = gene_map

    @cached_property
    # name -> gene IDs (given by gene_map.symbol_table)
    def _gene_ids(self) -> dict[str, tuple[list[int], list[int]]]:
        gene_ids = {}

        with open(self.gmt_path, "r") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 3:
                    continue

                name, direction = parts[0], parts[1]
                gene_names = parts[2:]

                if direction not in ["up", "down"]:
                    raise ValueError(
                        f"Expected 'up' or 'down' direction, got '{direction}'"
                    )

                if name not in gene_ids:
                    gene_ids[name] = ([], [])

                gene_list = []
                for gene_name in gene_names:
                    try:
                        gene_id = self.gene_map.symbol_table.index(gene_name)
                        gene_list.append(gene_id)
                    except KeyError:
                        raise ValueError(f"Gene '{gene_name}' not found in gene map")

                if direction == "up":
                    gene_ids[name] = (gene_list, gene_ids[name][1])
                else:  # direction == 'down'
                    gene_ids[name] = (gene_ids[name][0], gene_list)

        # Verify all names have both up and down
        for name, (up_genes, down_genes) in gene_ids.items():
            if not up_genes:
                raise ValueError(f"Missing 'up' entry for name '{name}'")
            if not down_genes:
                raise ValueError(f"Missing 'down' entry for name '{name}'")

        return gene_ids

    def gene_names(self, name: str, flag: str) -> list[str]:
        gids_up, gids_down = self._gene_ids[name]

        if flag == "up":
            return [self.gene_map.symbol_table[gid] for gid in gids_up]
        elif flag == "down":
            return [self.gene_map.symbol_table[gid] for gid in gids_down]
        elif flag == "both":
            return [self.gene_map.symbol_table[gid] for gid in gids_up + gids_down]
        else:
            raise ValueError(f"Flag must be 'up', 'down', or 'both', got '{flag}'")

    # name: name in a row like "name up gene1 gene2..."
    # flag: 'up', 'down', or 'both'
    def regions(self, name: str, flag: str) -> Tensor:
        gids_up, gids_down = self._gene_ids[name]

        intervals = []

        if flag == "up":
            for gid in gids_up:
                intervals.append(self.gene_map.regions[gid])
        elif flag == "down":
            for gid in gids_down:
                intervals.append(self.gene_map.regions[gid])
        elif flag == "both":
            for gid in gids_up + gids_down:
                intervals.append(self.gene_map.regions[gid])
        else:
            raise ValueError(f"Flag must be 'up', 'down', or 'both', got '{flag}'")

        if not intervals:
            # Return empty tensor with correct shape
            return torch.empty(0, 3, dtype=torch.int32)

        # Concatenate all tensors
        return torch.cat(intervals, dim=0)
