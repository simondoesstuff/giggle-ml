Some `bedtools` commands require a "genome file" (tsv that maps chrm -> size).

## Generate a bedtools "genome file"

```bash
fa='data/hg/hg38.fa'
out='data/hg/hg38_sizes.tsv'

samtools faidx $fa
cut -f 1,2 "${fa}.fai" > $out
rm "${fa}.fai"

# 455 data/hg/hg38_sizes.tsv
wc -l $out
```
