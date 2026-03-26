https://www.encodeproject.org/annotations/ENCSR842ULC/

Assumes you have `liftover`, `bgzip`, and `giggle`

```bash
chain='data/hg/hg19ToHg38.over.chain.gz'
[ ! -e $chain ] && echo "Missing chain file" && read
out='data/rheumatoid_arthritis'
mkdir -p $out

# acquire the data
wget -P $out https://www.encodeproject.org/files/ENCFF447UGJ/@@download/ENCFF447UGJ.bed.gz

# 1. filter for most significant
# 2. we only need the regions
zcat $out/ENCFF447UGJ.bed.gz | awk '$14 == "true" && $15 == "true" {print $1, $2, $3}' > $out/rheu-arth-hg19.bed

# perform liftover
liftover $out/rheu-arth-hg19.bed $chain $out/rheu-arth-hg38.bed $out/unmapped.bed

# cleanup
wc -l $out/*
#   13707126 data/rheumatoid_arthritis/rheu-arth-hg38.bed
echo -n "press Enter to cleanup..." && read
bgzip $out/rheu-arth-hg38.bed
rm $out/ENCFF447UGJ.bed.gz
rm $out/unmapped.bed
rm $out/rheu-arth-hg19.bed
```

Get `giggle` RME ranks

```bash
gindex='data/roadmap_epigenomics/beds.giggle'
giggle search -i $gindex -s -q $out/rheu-arth-hg38.bed.gz | tee $out/giggle_rme.tsv
```
