library(ggplot2)
library(viridis)
library(Seurat)
library(Rtsne)

source("functions.R")
fromScratch <- TRUE

load("../RData/QC.RData")

if (fromScratch){
    ## assign clusters by Seurat ##
    pbmc <- CreateSeuratObject(counts.log2cpm, meta.data = anno.qc)
    pbmc <- FindVariableFeatures(pbmc, selection.method='vst', nfeatures = 1500)  # if an error is returned, use selection.method='mean.var.plot'
    all.genes <- rownames(pbmc)
    pbmc <- ScaleData(pbmc, features = all.genes)
    pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))
    DimHeatmap(pbmc, dims = 1:10, cells = 500, balanced = TRUE)
    ElbowPlot(pbmc)
    pbmc <- FindNeighbors(pbmc, dims = 1:10)
    pbmc <- FindClusters(pbmc, resolution = 0.35)  
    anno.qc$seuratCluster <- as.numeric(Idents(pbmc))  # convert from 0:6 to 1:7

    # rename clusters
    id_from <- 1:7
    id_to <- c(
        "Naive T",
        "Myeloid 1",
        "NK",
        "CTL",
        "B",
        "Myeloid 2",
        "PC"
    )
    for (i in id_from){
        anno.qc$seuratCluster[anno.qc$seuratCluster == i] <- id_to[i]
    }
    anno.qc$seuratCluster <- factor(
        anno.qc$seuratCluster,
        levels=c(
            "Naive T",
            "CTL",
            "NK",
            "B",
            "PC",
            "Myeloid 1",
            "Myeloid 2"
        )
    )

    ## make t-SNE plots
    # remove genes expressed by less than 10 % of all cells
    expressed_genes <- apply(
        counts.log2cpm,
        1,
        function(x){
            sum(x != 0) / length(x) >= 0.05
        }
    )

    # 8956 genes survive
    counts.log2cpm <- counts.log2cpm[expressed_genes, ]

    # calculate variance and order the data by that
    num_genes_to_use <- 1500  # number of high-variance genes to be used
    variance <- apply(counts.log2cpm, 1, var)
    high_var <- names(
        variance[order(variance, decreasing=TRUE)][1:num_genes_to_use]
    )

    set.seed(0)
    t <- Rtsne(t(counts.log2cpm)[, rownames(counts.log2cpm) %in% high_var])

    d.tsne <- as.data.frame(t$Y)

    # save the result
    saveRDS(anno.qc, file="../RData/annotation_QCwithCluster.rds")
    saveRDS(d.tsne, file="../RData/tsne_QC-all.rds")
} else {
    anno.qc <- readRDS("../RData/annotation_QCwithCluster.rds")
    d.tsne <- readRDS("../RData/tsne_QC-all.rds")
}

counts.log2cpm <- as.data.frame(t(counts.log2cpm))

p.tsne <- plot.tsne(d.tsne, anno.qc, COLOR="seuratCluster", COLOR.LAB = "Cell type")
p.tsne.CD3E <- plot.tsne(d.tsne, anno.qc, COLOR="CD3E", GENE=TRUE, GENE.DF=counts.log2cpm)
p.tsne.CD3D <- plot.tsne(d.tsne, anno.qc, COLOR="CD3D", GENE=TRUE, GENE.DF=counts.log2cpm)
p.tsne.CD3G <- plot.tsne(d.tsne, anno.qc, COLOR="CD3G", GENE=TRUE, GENE.DF=counts.log2cpm)
p.tsne.CD3Z <- plot.tsne(d.tsne, anno.qc, COLOR="CD247", GENE=TRUE, GENE.DF=counts.log2cpm)
p.tsne.CD4 <- plot.tsne(d.tsne, anno.qc, COLOR="CD4", GENE=TRUE, GENE.DF=counts.log2cpm)
p.tsne.CD8A <- plot.tsne(d.tsne, anno.qc, COLOR="CD8A", GENE=TRUE, GENE.DF=counts.log2cpm)
p.tsne.CD8B <- plot.tsne(d.tsne, anno.qc, COLOR="CD8B", GENE=TRUE, GENE.DF=counts.log2cpm)
p.tsne.NK <- plot.tsne(d.tsne, anno.qc, COLOR="NCAM1", GENE=TRUE, GENE.DF=counts.log2cpm)
p.tsne.gzmb <- plot.tsne(d.tsne, anno.qc, COLOR="GZMB", GENE=TRUE, GENE.DF=counts.log2cpm)

p.vl.cd3d <- plot.violin(counts.log2cpm, anno.qc, Y="CD3D", X="seuratCluster", FILL="seuratCluster") + theme(legend.position = "none")
p.vl.gzmb <- plot.violin(counts.log2cpm, anno.qc, Y="GZMB", X="seuratCluster", FILL="seuratCluster") + theme(legend.position = "none")

pngStore(p.tsne, "../image/tsne.png", HEIGHT=5, WIDTH=6)
pngStore(p.tsne.CD3E, "../image/tsne_CD3E.png", HEIGHT=5, WIDTH=5.5)
pngStore(p.tsne.CD3D, "../image/tsne_CD3D.png", HEIGHT=5, WIDTH=5.5)
pngStore(p.tsne.CD3G, "../image/tsne_CD3G.png", HEIGHT=5, WIDTH=5.5)
pngStore(p.tsne.CD3Z, "../image/tsne_CD3Z.png", HEIGHT=5, WIDTH=5.5)
pngStore(p.tsne.CD4, "../image/tsne_CD4.png", HEIGHT=5, WIDTH=5.5)
pngStore(p.tsne.CD8A, "../image/tsne_CD8A.png", HEIGHT=5, WIDTH=5.5)
pngStore(p.tsne.CD8B, "../image/tsne_CD8B.png", HEIGHT=5, WIDTH=5.5)
pngStore(p.tsne.NK, "../image/tsne_NCAM1.png", HEIGHT=5, WIDTH=5.5)
pngStore(p.tsne.gzmb, "../image/tsne_gzmb.png", HEIGHT=5, WIDTH=5.5)
pngStore(p.vl.cd3d, "../image/vl_cd3d.png", HEIGHT=4, WIDTH=5.5)
pngStore(p.vl.gzmb, "../image/vl_gzmb.png", HEIGHT=4, WIDTH=5.5)


# some additional plots
T <- anno.qc$seuratCluster %in% c(0, 3)
counts.log2cpm$NKmean <- (counts.log2cpm$NCAM1 + counts.log2cpm$KLRB1) / 2
counts.log2cpm$CD48DN <- (counts.log2cpm$CD4 == 0) & ((counts.log2cpm$CD8A == 0) & (counts.log2cpm$CD8B == 0))
counts.log2cpm$CD48DN <- gsub(TRUE, "DN", counts.log2cpm$CD48DN)
counts.log2cpm$CD48DN <- gsub(FALSE, "Positive", counts.log2cpm$CD48DN)
counts.log2cpm$CD48DN <- factor(counts.log2cpm$CD48DN, levels=c("Positive", "DN"))
p.tsne.NK2 <- plot.tsne(d.tsne[T, ], anno.qc[T, ], COLOR="KLRB1", GENE=TRUE, GENE.DF=counts.log2cpm[T, ])
p.tsne.NKmean <- plot.tsne(d.tsne[T, ], anno.qc[T, ], COLOR="NKmean", GENE=TRUE, GENE.DF=counts.log2cpm[T, ]) + labs(color="(NCAM1 + KLRB1) / 2")
p.tsne.DN <- plot.tsne(d.tsne[T, ], anno.qc[T, ], COLOR="CD48DN", GENE=TRUE, GENE.DF=counts.log2cpm[T, ]) 
p.tsne.DN <- p.tsne.DN + labs(color="CD4/8") + scale_color_viridis_d()
p.tsne.HIF1A <- plot.tsne(d.tsne[T, ], anno.qc[T, ], COLOR="ARNT", GENE=TRUE, GENE.DF=counts.log2cpm[T, ])


# save analysis
#save.image(file=file.path("../RData", paste0("DNT_clustering-tSNE", "_", Sys.Date())))