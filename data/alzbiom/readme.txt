AlzBiom
======

Gut microbiome of Alzheimer's disease (AD)

Publication:
- Laske C, Müller S, Preische O, et al. Signature of Alzheimer’s Disease in
  Intestinal Microbiome: Results From the AlzBiom Study. Front Neurosci. 2022.
  16:792996. doi: 10.3389/fnins.2022.792996
- https://www.frontiersin.org/articles/10.3389/fnins.2022.792996/full

Data release (ENA):
- https://www.ebi.ac.uk/ena/browser/view/PRJEB47976
- These should be clean (QC'ed, host-filtered) sequencing data.

Note:
- The paper says: "Human contamination was removed by mapping reads against the
  human genome (GRCh38) using KneadData." It is not clear whether these FASTQ
  files are post this step. A Bowtie2 alignment of one sample against the human
  reference genome yielded 0.12% alignment rate, which is lower than typical
  raw gut metagenome, but still not fully clean.

Code release:
- https://github.com/UliSchopp/AlzBiom
- Contains metadata

Cohort (n = 175):
- Amyloid-positive AD patients (AD = 1) (n = 75)
- Cognitively healthy controls (AD = 0) (n = 100)

Note:
- Meanwhile, APOE4 is a strong clinical predictor of AD.

Alignments:
- /mnt/scratch0/drdr/alzbiom-woltka-2/aln

Aligned against WoLr2 using Bowtie2 v2.4.5.

```
db=/mnt/store0/qiyun/wol2/databases/bowtie2/WoLr2
while read id; do
  bowtie2 -p 8 -x $db -1 fastq/${id}_1.fastq.gz -2 fastq/${id}_2.fastq.gz \
    --very-sensitive --no-head --no-unal | cut -f1-9 | sed 's/$/\t*\t*/' |\
    xz -9 > align/$id.sam.xz
done < ids.txt
```

Processed using Woltka.

```
# OGU table:
woltka classify -i align -o ogu.biom

# ORF table (unit: RPK):
coords=/mnt/store0/qiyun/wol2
woltka classify -i align -o orf.biom -c $coords \
  --sizes . --scale 1k --digits 3
```

Data tables:
- /mnt/scratch0/drdr/alzbiom-woltka-2
