#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 10:17:09 2021

@author: abrahamr
"""
# Deep dive into large CD4neg aggregates 
# Visualize "large CD4neg" aggregates and make sure that you are capturing the right population
# Within these aggregates, what percentage of cells are Tfh?
# What are the nearest neighbors relationships?

import os, sys
import numpy as np
import pandas as pd 
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tifffile import imread,imsave
from skimage.draw import rectangle_perimeter
from scipy.stats import ks_2samp, mannwhitneyu,pearsonr
from NeighborhoodAnalysis import *
from AnalyzeSecondaries import *
from TotalPieChart import *
def get_percluster_counts_sub(b,all_cells_df,s):
    df=all_cells_df.groupby(['clusters','Subset_id']).count()
    df=df.reset_index()
    df=df.pivot(index='clusters',columns='Subset_id',values='cell_id')
    df=df.replace(np.NaN, 0)
    for x in range(1,9):
        if x not in df.columns:
            df[x]=np.repeat(0,df.shape[0])
    df=df[list(range(1,9))]       
    df=df.reset_index()
    df.columns=['clusters','DN','CD4+','CD8+', 'CD20+','CD138+','Treg','Tfh','exhausted_CD8+']
    df['total_CD4+']=sum([df['CD4+'],df['Tfh'],df['Treg']])
    df['total_CD8+']=sum([df['CD8+'],df['exhausted_CD8+']])
    df['total_cells']=sum([df['DN'],df['CD4+'],df['CD8+'],
                                   df['CD20+'],df['CD138+'],
                                   df['Treg'],df['Tfh'],df['exhausted_CD8+']])
    df['Accession #']=b
    df['section']=s
    return df

def ExtractfromClin(clin_df,add_df,feat,featname):
    feat_dict = dict(zip(clin_df['Accession #'],clin_df[feat]))
    def check_feat(Acc):
        return feat_dict[Acc]
    add_df[featname] = add_df['Accession #'].apply(check_feat)
    return add_df 

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvdir", default= "CellFeats_csvs", help = "path to directory with data files (csvs)") 
    parser.add_argument("--svdr",default="CD4neg_Clusters/", help = "path to save directory")
    parser.add_argument("--outfile",default="CD4neg_clusters.txt", help = "text file to save outputs to")
    parser.add_argument("--compdir",default="cell_composite", help = "directory with stitched cell composites")
    args = parser.parse_args()
    savedir=args.svdr
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    csv_dir=args.csvdir
   
    outfile=os.path.join(savedir,args.outfile)

    stdoutOrigin=sys.stdout
    sys.stdout = open(str(outfile),'w')
    # Define directories to pull from
    csvs=os.listdir(csv_dir)
    csvs=[x for x in csvs if x.endswith('.csv')]
    csvs.sort()
    biopsy_list=[x.split('_')[0] for x in csvs]
    biopsy_list=list(np.unique(biopsy_list))

    imdir=args.compdir
    
    #Get cluster counts for all biopsies
    cluster_count_list=[]
    cells_list=[]

    for c in csvs:
        b=c.split('_')[0]
        s='_'.join(c.split('_')[:-1])
        df=pd.read_csv(os.path.join(csv_dir,c))
        cluster_counts_s=get_percluster_counts_sub(b,df,s)
        cluster_counts_s=cluster_counts_s.loc[cluster_counts_s['clusters']!=-1] #remove singlets
        #Identify B-T clusters
        clusters_of_interest=list(cluster_counts_s.loc[(((cluster_counts_s['total_CD8+']+cluster_counts_s['DN'])/cluster_counts_s['total_cells'])>0.25)
                                                       &(cluster_counts_s['total_cells']<20)]['clusters'])
        
        im_arr=imread(os.path.join(imdir,s+'_cells_stitched.tif'))
        COI=[]
        for ind,row in df.iterrows():
            if row['clusters'] in clusters_of_interest:
                COI.append(1)
            else:
                COI.append(0)
        df['COI']=COI
        df['Accession #']=b
        if len(im_arr.shape)==3:
            for cl in clusters_of_interest:
                cluster=df.loc[df['clusters']==cl]
                min_row=int(min(cluster['global_centroid_row'])-20)
                max_row=int(max(cluster['global_centroid_row'])+20)
                min_col=int(min(cluster['global_centroid_column'])-20)
                max_col=int(max(cluster['global_centroid_column'])+20)
                BB=rectangle_perimeter((min_row,min_col),(max_row,max_col))
                im_arr[BB[0],BB[1],0]=255
                im_arr[BB[0],BB[1],1]=255
                im_arr[BB[0],BB[1],2]=255
            #imsave(os.path.join(savedir,s+'_cells_stitched.tif'),im_arr)
        CD4neg_cells=df.loc[df['COI']==1]
       
        cluster_count_list.append(cluster_counts_s)
        cells_list.append(CD4neg_cells)
    cluster_counts_all=pd.concat(cluster_count_list,axis=0)
    CD4neg_cells=pd.concat(cells_list,axis=0)
   
    cluster_counts_all['Tex_prop']=cluster_counts_all['exhausted_CD8+']/cluster_counts_all['total_CD8+']
    cluster_counts_all['CD4neg_clusters']=((((cluster_counts_all['total_CD8+']+cluster_counts_all['DN'])/cluster_counts_all['total_cells'])>0.25)&(cluster_counts_all['total_cells']<20)).astype(int)
    print('Num CD4neg aggs vs non-CD4neg aggs')
    print(cluster_counts_all.groupby('CD4neg_clusters').count()['total_cells'])
    CD4neg_aggs=cluster_counts_all.loc[cluster_counts_all['CD4neg_clusters']==1]
 
    #Check overall distributions of cell classes
    pie_color_list=['#0bb81f','#b80b0b','#0425de', '#ba04de','#04cfde']
    CellCounts(CD4neg_cells, savedir, 'Distribution of cell subsets','BaseDist',pie_color_list)
    
    fig = plt.figure(figsize=(5,5), frameon=False)
    ax=fig.add_axes([0.25,0.1,0.75,0.85])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.scatter(CD4neg_aggs['total_CD8+'],CD4neg_aggs['DN'],s=CD4neg_aggs['total_cells'],color='black',alpha=0.4)
    plt.xlabel('CD8+ T cells')
    plt.ylabel('DN T cells')
    plt.savefig(savedir+'CD8_vs_DN.png')
    plt.show()
    R,p=pearsonr(CD4neg_aggs['total_CD8+'],CD4neg_aggs['DN'])
    
    CD4neg_aggs['DN_CD8_ratio']=(CD4neg_aggs['DN']+1)/(CD4neg_aggs['total_CD8+']+1)
    fig = plt.figure(figsize=(5,5), frameon=False)
    ax=fig.add_axes([0.25,0.1,0.75,0.85])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.hist(CD4neg_aggs['DN_CD8_ratio'],color='black', bins=50)
    plt.xlabel('DN:CD8 Ratio')
    plt.ylabel('Number of Neighborhoods')
    plt.savefig(savedir+'DN_CD8_ratio_hist.png')
    plt.show()
    
    # Get overall breakdown of T cell subsets
    cd4=CD4neg_cells.loc[CD4neg_cells['class_id']==2]
    cd4=get_allTsubsets(cd4)
    cd4sub=CellSubsets(cd4, savedir, 'Distribution of CD4 T cell subsets','CD4TSecondary_CD4negaggs')
    cd8=CD4neg_cells.loc[CD4neg_cells['class_id']==3]
    cd8=get_allTsubsets(cd8)
    cd8sub=CellSubsets(cd8, savedir, 'Distribution of CD8 T cell subsets','CD8TSecondary_CD4negaggs')
    dn=CD4neg_cells.loc[CD4neg_cells['class_id']==1]
    dn=get_allTsubsets(dn)
    dnsub=CellSubsets(dn, savedir, 'Distribution of DN T cell subsets','CD8TSecondary_DNnegaggs')
    sys.stdout.close()
    sys.stdout=stdoutOrigin
    
if __name__=='__main__':
    main()