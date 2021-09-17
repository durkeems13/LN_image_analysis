Post-segmentation analysis pipeline

Pre-processing 
# get_tile_positions.py - relates the tile number to the position in the whole biopsy; allows for the calculation of the global cell coords -> saves to "tile_positions.pkl"

Cellular Analyses (in order of operations)
# make_biopsy-level_csv.py - reads in "cleaned" prediction pickles and outputs a csv with all of the cells for a given sample -> saves to "CellFeats_csvs*" (specific name might vary) 
# HighDim_DBSCAN.py - uses sample csv to apply dbscan to the whole-composite; overwrites original csvs with new csvs with each cell assigned to a cluster, and composites of each cell colored by cluster -> saves to "CellFeats_csvs*"    
# ClusterAnalysis.py - analyzes clusters in terms of size and constituency -> saves to ClusterAnalysis
# AnalyzeSecondaries.py - generates pie charts of all the T cell subsets, broken down by base class -> all possible combinations of secondary markers -> saves to AnalyzeSecondaries 
# NeighborhoodAnalysis.py - niche analysis, nearest neighbors, and aggregation score calculation -> NeighborhoodAnalysis
# B-T_Clusters.py - subanalysis of large B-T neighborhoods -> saves to B-T_Clusters
# CD4neg_Clusters.py - subanalysis of CD4- neighborhoods -> saves to CD4neg_Clusters

 
Figure generation only
# TotalPieChart.py - generates pie charts of all cells, and T cell base subsets 

