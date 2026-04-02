These experiments concern heatmaps over RME.  
Similarity scores by `giggle` and CModel are required.

## Variations

based on the intervals post-liftover (hg38)

### \_random

`bedtools shuffle -seed 42 -chrom`

enforcing same chromosomes so we avoid introducing non-standard chromosomes

### \_expanded

`bedtools slop -b 50`

### \_masked

nucletodies entirely masked out; uses same artifacts as the base experiment, but new CModel embeddings with flag `-seq-dropout 1`
