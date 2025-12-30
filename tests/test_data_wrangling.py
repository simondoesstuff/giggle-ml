import numpy as np
import torch
from pathlib import Path

from giggleml.data_wrangling import fasta
from giggleml.data_wrangling.interval_dataset import BedDataset, MemoryIntervalDataset
from giggleml.data_wrangling.list_dataset import ListDataset
from giggleml.data_wrangling.unified_dataset import UnifiedDataset


def test_bed_fasta_parsing():
    bed = BedDataset("tests/test.bed", "tests/test.fa")
    assert bed[0] == ("chr1", 0, 1)
    assert bed[1] == ("chr1", 0, 40)
    assert bed[4] == ("chr1", 15, 25)
    assert bed[5] == ("chr1", 10, 30)
    assert bed[6] == ("chr2", 10, 30)
    assert bed[-1] == ("chr3", 5, 10)
    assert bed[-2] == ("chr3", 0, 5)

    fa = fasta.map(bed)
    assert len(fa) == len(bed)
    assert fa[-1] == "TTTTT"
    assert fa[-2] == "AAAAA"


# def testSlidingWindowDataset():
#     # TODO: SlidingWindowDataset requires further tests.
#     # there's weird stuff going on
#
#     content = [
#         "Steaks are best hot!",
#         "123456789",
#     ]
#     dataset = SlidingWindowDataset(content, 0.5)
#     expecting = [
#         "Steaks are",
#         "s are best",
#         " best hot!",
#         "1234",
#         "3456",
#         "5678",
#     ]
#
#     for i, expect in enumerate(expecting):
#         assert expect == dataset[i]


def test_unified_dataset():
    sizes = [5, 20, 8, 13]
    lists = [list(range(i)) for i in sizes]
    datasets = [ListDataset(items) for items in lists]
    uni_set = UnifiedDataset[int](datasets)
    expect = np.concatenate(lists)

    assert len(uni_set) == len(expect)

    for i, item in enumerate(expect):
        assert item == uni_set[i]

    np.random.seed(42)
    walk = list(range(len(expect)))
    np.random.shuffle(walk)

    for i in walk:
        # tests random access
        assert expect[i] == uni_set[i]


def test_to_gpu_serializable_with_dict():
    """Test to_gpu_serializable with fasta dict input"""
    fasta_dict = {'chr1': 'ATCG', 'chr2': 'GCTA'}
    
    def tokenizer(seq):
        mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        return torch.tensor([mapping[c] for c in seq], dtype=torch.long)
    
    tokens, offsets, sizes, chr_to_idx = fasta.to_gpu_serializable(fasta_dict, tokenizer)
    
    # Check shapes and types
    assert isinstance(tokens, torch.Tensor)
    assert isinstance(offsets, torch.Tensor)
    assert isinstance(sizes, torch.Tensor)
    assert isinstance(chr_to_idx, dict)
    
    # Check values
    expected_tokens = torch.tensor([0, 1, 2, 3, 3, 2, 1, 0], dtype=torch.long)
    expected_offsets = torch.tensor([0, 4], dtype=torch.long)
    expected_sizes = torch.tensor([4, 4], dtype=torch.long)
    expected_chr_to_idx = {'chr1': 0, 'chr2': 1}
    
    assert torch.equal(tokens, expected_tokens)
    assert torch.equal(offsets, expected_offsets)
    assert torch.equal(sizes, expected_sizes)
    assert chr_to_idx == expected_chr_to_idx


def test_to_gpu_serializable_with_path():
    """Test to_gpu_serializable with file path input"""
    def tokenizer(seq):
        mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        return torch.tensor([mapping[c] for c in seq], dtype=torch.long)
    
    # Test with string path
    tokens_str, offsets_str, sizes_str, chr_to_idx_str = fasta.to_gpu_serializable("tests/test.fa", tokenizer)
    
    # Test with Path object
    path_obj = Path("tests/test.fa")
    tokens_path, offsets_path, sizes_path, chr_to_idx_path = fasta.to_gpu_serializable(path_obj, tokenizer)
    
    # Check that both approaches produce equivalent results
    assert torch.equal(tokens_str, tokens_path)
    assert torch.equal(offsets_str, offsets_path)
    assert torch.equal(sizes_str, sizes_path)
    assert chr_to_idx_str == chr_to_idx_path
    
    # Check basic properties
    assert isinstance(tokens_str, torch.Tensor)
    assert len(chr_to_idx_str) > 0  # Should have chromosomes
    assert len(offsets_str) == len(sizes_str)  # Should be aligned
    assert len(offsets_str) == len(chr_to_idx_str)  # One per chromosome


def test_bed_dataset_with_extra_columns():
    """Test BED dataset parsing with additional columns (like gene names)"""
    # Create test BED data with gene names in column 4
    test_bed_content = """chr1\t1000\t2000\tGENE1
chr2\t3000\t4000\tGENE2
chr3\t5000\t6000\tGENE3\textra_info
chrX\t7000\t8000\tGENEX"""
    
    # Write test file
    test_bed_path = Path("tests/test_with_genes.bed")
    test_bed_path.write_text(test_bed_content)
    
    try:
        # Test BedDataset
        bed = BedDataset(test_bed_path)
        assert len(bed) == 4
        
        # Test backward compatibility - standard iteration should still work
        intervals = list(bed)
        assert intervals[0] == ("chr1", 1000, 2000)
        assert intervals[1] == ("chr2", 3000, 4000)
        assert intervals[2] == ("chr3", 5000, 6000)
        assert intervals[3] == ("chrX", 7000, 8000)
        
        # Test new functionality - iteration with extra columns
        intervals_with_extras = list(bed.iter_with_extra_columns())
        assert len(intervals_with_extras) == 4
        
        interval1, extras1 = intervals_with_extras[0]
        assert interval1 == ("chr1", 1000, 2000)
        assert extras1 == ["GENE1"]
        
        interval2, extras2 = intervals_with_extras[1]
        assert interval2 == ("chr2", 3000, 4000)
        assert extras2 == ["GENE2"]
        
        interval3, extras3 = intervals_with_extras[2]
        assert interval3 == ("chr3", 5000, 6000)
        assert extras3 == ["GENE3", "extra_info"]  # Multiple extra columns
        
        interval4, extras4 = intervals_with_extras[3]
        assert interval4 == ("chrX", 7000, 8000)
        assert extras4 == ["GENEX"]
        
    finally:
        # Clean up test file
        if test_bed_path.exists():
            test_bed_path.unlink()


def test_memory_interval_dataset_with_extra_columns():
    """Test MemoryIntervalDataset with extra columns functionality"""
    intervals = [("chr1", 100, 200), ("chr2", 300, 400)]
    extra_columns = [["GENE_A"], ["GENE_B", "annotation"]]
    
    dataset = MemoryIntervalDataset(intervals, extra_columns=extra_columns)
    
    # Test backward compatibility
    assert len(dataset) == 2
    assert dataset[0] == ("chr1", 100, 200)
    assert dataset[1] == ("chr2", 300, 400)
    
    # Test standard iteration
    intervals_iter = list(dataset)
    assert intervals_iter == [("chr1", 100, 200), ("chr2", 300, 400)]
    
    # Test iteration with extra columns
    with_extras = list(dataset.iter_with_extra_columns())
    assert len(with_extras) == 2
    
    interval1, extras1 = with_extras[0]
    assert interval1 == ("chr1", 100, 200)
    assert extras1 == ["GENE_A"]
    
    interval2, extras2 = with_extras[1]
    assert interval2 == ("chr2", 300, 400)
    assert extras2 == ["GENE_B", "annotation"]
