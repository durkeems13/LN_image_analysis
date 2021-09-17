library(ggplot2)
library(viridis)
library(ggridges)

source("functions.R")

load("../RData/QC.RData")
anno <- readRDS("../RData/annotation_QCwithCluster.rds")

# select T cells based on Seurat clusters
T <- anno$seuratCluster %in% c("Naive T", "CTL")
counts.t <- counts.qc[, T]
counts.log2cpm.t <- counts.log2cpm[, T]
anno.t <- anno[T, ]


# label by CD4/CD8 expression
CD4 <- counts.log2cpm.t["CD4", ] > 0
CD8A <- counts.log2cpm.t["CD8A", ] > 0
CD8B <- counts.log2cpm.t["CD8B", ] > 0
cd4_8.freq <- rep("DN", sum(T))
cd4_8.freq[CD4] <- "CD4" 
cd4_8.freq[CD8A | CD8B] <- "CD8" 
cd4_8.freq[CD4 & (CD8A | CD8B)] <- "DP" 
anno.t$CD4.8 <- factor(cd4_8.freq, levels=c("DP", "CD4", "CD8", "DN"))


add_column <- function(df, colname){
    col <- counts.log2cpm.t[colname, ] > 0
    col <- gsub(TRUE, "Positive", col)
    col <- gsub(FALSE, "Negative", col)
    df[, colname] <- factor(col, levels=c("Positive", "Negative"))

    return(df)
}


anno.t <- add_column(anno.t, "CD3D")
anno.t <- add_column(anno.t, "CD3E")
anno.t <- add_column(anno.t, "CD3G")
anno.t <- add_column(anno.t, "CD247")
p.pie.cd4.8 <- plot.pie(anno.t, "CD4.8", "CD4/8 expression") + scale_fill_viridis_d() 
p.pie.cd3d <- plot.pie(anno.t, "CD3D", "CD3D expression") + scale_fill_viridis_d(option="plasma")
p.pie.cd3e <- plot.pie(anno.t, "CD3E", "CD3E expression") + scale_fill_viridis_d(option="plasma")
p.pie.cd3g <- plot.pie(anno.t, "CD3G", "CD3G expression") + scale_fill_viridis_d(option="plasma")
p.pie.cd3z <- plot.pie(anno.t, "CD247", "CD247 expression") + scale_fill_viridis_d(option="plasma")
pngStore(p.pie.cd4.8, "../image/cd4-8.png", WIDTH=4, HEIGHT=3)
pngStore(p.pie.cd3d, "../image/cd3d.png", WIDTH=4, HEIGHT=3)
pngStore(p.pie.cd3e, "../image/cd3e.png", WIDTH=4, HEIGHT=3)
pngStore(p.pie.cd3g, "../image/cd3g.png", WIDTH=4, HEIGHT=3)
pngStore(p.pie.cd3z, "../image/cd3z.png", WIDTH=4, HEIGHT=3)


# analyze TRAC/TRDC expression
# make a data frame for plotting density plots
d.trac_trdc <- as.data.frame(t(counts.log2cpm.t))

d.trac_trdc$CD4.8 <- anno.t$CD4.8
# scatter of TRAC/TRDC
p <- ggplot(d.trac_trdc, aes(x=TRAC, y=TRDC, color=CD4.8)) + geom_point() 
p <- p + scale_color_viridis_d() + theme_classic() + labs(color="CD4/8 expression")
p.trac_trdc_scatter <- p + theme(axis.title = element_text(face = "italic"))
pngStore(p.trac_trdc_scatter, "../image/scatter_TRAC-TRDC.png", WIDTH=5, HEIGHT=3)


plot_ridge <- function(
    X, XLAB, 
    data=d.trac_trdc, Y="CD4.8", YLAB="CD4/8 expression", 
    FILL="CD4.8", ALPHA=0.8
){
    p <- ggplot(data, aes(x=data[, X], y=data[, Y], fill=data[, FILL])) 
    p <- p + geom_density_ridges(alpha=ALPHA) 
    p <- p + scale_fill_viridis_d() + ylab("") + xlab(XLAB) + guides(fill=FALSE)
    p <- p + theme_ridges() + theme_classic() + theme(
        axis.title.x = element_text(hjust = 0.5, face = "italic"),
        axis.title.y = element_text(hjust = 0.5),
        axis.text = element_text(color="black")
    )
    p <- p + coord_cartesian(clip = "off")  # prevent the plot from being trimmed
    return(p)
}


p.trac <- plot_ridge(X="TRAC", XLAB="TRAC") 
p.trdc <- plot_ridge(X="TRDC", XLAB="TRDC") 
p.ncam1 <- plot_ridge(X="NCAM1", XLAB="NCAM1") 
p.klrb1 <- plot_ridge(X="KLRB1", XLAB="KLRB1") 
p.trav1.2 <- plot_ridge(X="TRAV1-2", XLAB="TRAV1-2") 
pngStore(p.trac, "../image/ridge_trac.png", WIDTH=4, HEIGHT=3)
pngStore(p.trdc, "../image/ridge_trdc.png", WIDTH=4, HEIGHT=3)
pngStore(p.ncam1, "../image/ridge_ncam1.png", WIDTH=4, HEIGHT=3)
pngStore(p.klrb1, "../image/ridge_klrb1.png", WIDTH=4, HEIGHT=3)

# visualize CD4 and CD8A raw count distribution
raw_df <- as.data.frame(t(counts.qc[c("CD4", "CD8A", "CD8B"), T]))
p.raw.cd4 <- ggplot(raw_df, aes(x=CD4)) + geom_histogram(binwidth=1) + theme_classic() + xlab(expression(paste(italic("CD4"), " ", "UMI")))
p.raw.cd8a <- ggplot(raw_df, aes(x=CD8A)) + geom_histogram(binwidth=1) + theme_classic() + xlab(expression(paste(italic("CD8A"), " ", "UMI")))
p.raw.cd8b <- ggplot(raw_df, aes(x=CD8B)) + geom_histogram(binwidth=1) + theme_classic() + xlab(expression(paste(italic("CD8B"), " ", "UMI")))
pngStore(p.raw.cd4, "../image/umi_cd4.png", WIDTH=4, HEIGHT=2.5)
pngStore(p.raw.cd8a, "../image/umi_cd8a.png", WIDTH=4, HEIGHT=2.5)
pngStore(p.raw.cd8b, "../image/umi_cd8b.png", WIDTH=4, HEIGHT=2.5)

# calculate ratio of TRAC and TRDC-expressing cells per CD4/8 category
print(dplyr::group_by(d.trac_trdc, CD4.8) %>% dplyr::summarize(sum(TRAC > 0) / length(TRAC)))
print(dplyr::group_by(d.trac_trdc, CD4.8) %>% dplyr::summarize(sum(TRDC > 0) / length(TRDC)))

# save analysis
#save.image(file=file.path("../RData", paste0("DNT_postClustering_gdTCRanalysis", "_", Sys.Date())))