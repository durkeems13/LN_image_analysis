# packages
library(ggplot2)
library(edgeR)
library(Seurat)

source("functions.R")

# Load gene expression data.
counts <- read.csv(
    "../input/SDY997_EXP15176_celseq_matrix_ru10_molecules.tsv", 
    row.names = 1, 
    sep = "\t"
)  # gene x cell

anno <- read.csv("../input/SDY997_EXP15176_celseq_meta.tsv", sep = "\t")


# Subset the Lupus data.
counts <- counts[, (anno$sample != "none") & (anno$disease == "SLE")]
anno <- anno[(anno$sample != "none") & (anno$disease == "SLE"), ]

# Subset Leukocytes
counts <- counts[, anno$type == "Leukocyte"]
anno <- anno[anno$type == "Leukocyte", ]

# Remove cells with zero counts for endogenous genes.
notZeroCounts <- colSums(counts) != 0
counts <- counts[, notZeroCounts]
anno <- anno[notZeroCounts, ]


## QC the data
# Count the number of expressed genes.
geneNum <- colSums(counts != 0)
anno$gene.number <- geneNum

# Grep mitochondrial genes
MT <- grep("MT-", rownames(counts))

# Calculate MT-gene ratio (%).
MTratio <- colSums(counts[MT, ]) / colSums(counts)
anno$MTratio <- MTratio

# filter the data by qc criteria used by the authors
qc.geneNum.min <- 1000
qc.geneNum.max <- 5000
qc.MTratio <- 0.25
# also remove cells with extremely high UMI counts (top 1%)
#qc.umi <- quantile(anno$molecules, 0.99)
#qc <- (MTratio < qc.MTratio) & ((geneNum >= qc.geneNum) & (anno$molecules <= qc.umi))
qc <- (MTratio < qc.MTratio) & ((geneNum >= qc.geneNum.min) & (geneNum < qc.geneNum.max))

# plot sequencing depth information before QC
p.reads <- ggplot(anno, aes(x=reads_ru1)) + geom_histogram() # total mapped reads 
p.umi <- ggplot(anno, aes(x=molecules)) + geom_histogram() # filtered umi
p.gene_mt <- ggplot(anno, aes(x=MTratio, y=gene.number)) + geom_point() + xlab("MT-gene ratio") + ylab("Detected gene count")
pngStore(p.reads, "../image/SLE-Leukocyte-beforeQC_reads-ru1.png", WIDTH=4, HEIGHT=3)
pngStore(p.umi, "../image/SLE-Leukocyte-beforeQC_umi.png", WIDTH=4, HEIGHT=3)
pngStore(p.gene_mt, "../image/SLE-Leukocyte-beforeQC_geneNum-MT.png", WIDTH=4, HEIGHT=4)

# subset by QC criteria
counts.qc <- counts[, qc]
anno.qc <- anno[qc, ]

# normalize log2-cpm
counts.log2cpm <- log2(cpm(counts.qc) + 1) 

# save analysis
save(counts.log2cpm, counts.qc, anno.qc, file="../RData/QC.RData")
#save.image(file=file.path("../RData", paste0("DNT_QC", "_", Sys.Date())))
