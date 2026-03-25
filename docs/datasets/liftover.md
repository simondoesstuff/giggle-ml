How to do a genomic "lift over" (to new assemblies):

## Install LiftOver

For nix users: `nix develop`

Or follow the guide: https://hgdownload.gi.ucsc.edu/downloads.html#utilities_downloads

### hg19 to hg38

```bash
mkdir -p data/hg
# get the chain file
wget -P data/hg https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz
```
