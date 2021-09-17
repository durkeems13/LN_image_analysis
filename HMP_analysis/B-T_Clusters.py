#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 10:17:09 2021

@author: abrahamr
"""
# Deep dive into large T-B aggregates 
# Visualize "large T-B" aggregates and make sure that you are capturing the right population
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
from scipy.stats import ks_2samp, mannwhitneyu
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
    parser.add_argument("--svdr",default='B-T_Clusters/', help = "path to save directory")
    parser.add_argument("--outfile",default="BT_clusters.txt", help = "text file to save outputs to")
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
    csv_dir='CellFeats_csvs_20210616/'
    csvs=os.listdir(csv_dir)
    csvs=[x for x in csvs if x.endswith('.csv')]
    csvs.sort()
    biopsy_list=[x.split('_')[0] for x in csvs]
    biopsy_list=list(np.unique(biopsy_list))
    imdir=args.compdir
    # Class definitions
    class_dict={1:'DN',2:'CD4+',3:'CD8+',4:'CD20+',5:'CD138+',6:'Treg',7:'Tfh',8:'exhausted_CD8+'}
    class_list=['DN','CD4+','CD8+','CD20+','CD138+','Treg','Tfh','exhausted_CD8+']
    color_dict={'DN':'#0bb81f','CD4+':'#b80b0b','CD8+':'#0425de','CD20+':'#ba04de',
                    'CD138+':'#04cfde','Treg':'#ba73a6','Tfh':'#91193d','exhausted_CD8+':'#5f60b3'}
    c_list=[color_dict[x] for x in class_list]
    
    #Get cluster counts for all biopsies
    cluster_count_list=[]
    cells_list=[]
    NN=[]
    for c in csvs:
        b=c.split('_')[0]
        s='_'.join(c.split('_')[:-1])
        df=pd.read_csv(os.path.join(csv_dir,c))
        cluster_counts_s=get_percluster_counts_sub(b,df,s)
        cluster_counts_s=cluster_counts_s.loc[cluster_counts_s['clusters']!=-1] #remove singlets
        #Identify B-T clusters
        clusters_of_interest=list(cluster_counts_s.loc[(((cluster_counts_s['total_CD4+']+cluster_counts_s['CD20+'])/cluster_counts_s['total_cells'])>0.5)
                                 &(cluster_counts_s['total_cells']>20)&(cluster_counts_s['CD20+']>0)&(cluster_counts_s['total_CD4+']>0)]['clusters'])
        
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
            imsave(os.path.join(savedir,s+'_cells_stitched.tif'),im_arr)
        BT_cells=df.loc[df['COI']==1]
        
        if BT_cells.shape[0]>0:
            nn_agg=CheckMinDist_global(BT_cells,s)
            NN.append(nn_agg)
            
        
        cluster_count_list.append(cluster_counts_s)
        cells_list.append(BT_cells)
    cluster_counts_all=pd.concat(cluster_count_list,axis=0)
    BT_cells=pd.concat(cells_list,axis=0)
   
    cluster_counts_all['Tfh_prop']=cluster_counts_all['Tfh']/cluster_counts_all['total_CD4+']
    cluster_counts_all['B-T']=((((cluster_counts_all['total_CD4+']+cluster_counts_all['CD20+'])/cluster_counts_all['total_cells'])>0.5)
                                 &(cluster_counts_all['total_cells']>20)&(cluster_counts_all['CD20+']>0)&(cluster_counts_all['total_CD4+']>0)).astype(int)
    print('Num B-T aggs vs non-BT aggs')
    print(cluster_counts_all.groupby('B-T').count()['total_cells'])
    BT_aggs=cluster_counts_all.loc[cluster_counts_all['B-T']==1]
    non_BT_aggs=cluster_counts_all.loc[cluster_counts_all['B-T']==0]
    textargs={'size':'16','fontname':'Times New Roman'}
    medianlineprops={'linewidth':'0'}
    boxprops={'linewidth':'2'}
    whiskerprops={'linewidth':'2'}
    capprops={'linewidth':'2'}
    meanprops=dict(markerfacecolor='white',markeredgecolor='black',marker='D',markersize=5)
    plt.clf()
    fig = plt.figure(figsize=(5,5), frameon=False)
    ax=fig.add_axes([0.25,0.1,0.75,0.85])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    bplot=ax.boxplot([non_BT_aggs['Tfh_prop'].dropna(),BT_aggs['Tfh_prop']],labels=['non B-T clusters','B-T clusters'], 
                      patch_artist=True, showmeans=True, widths=0.4,boxprops=boxprops,
                      whiskerprops=whiskerprops,medianprops=medianlineprops,
                      capprops=capprops,meanprops=meanprops)
    for patch, color in zip(bplot['boxes'],['blue','green']):
        patch.set_facecolor(color)
    plt.ylabel('Proportion of CD4+ T cells that are Tfh',**textargs)
    plt.xticks(fontsize=14,fontname='Times New Roman')
    plt.yticks(fontsize=14,fontname='Times New Roman')
    plt.savefig(savedir+'B-T_agg_tfh_prop.png')
    plt.show()
    
    u,p=mannwhitneyu(BT_aggs['Tfh_prop'],non_BT_aggs['Tfh_prop'])
    print('Difference between BT aggs and non B-T aggs: '+str(p))
    #Check overall distributions of cell classes
    pie_color_list=['#0bb81f','#b80b0b','#0425de', '#ba04de','#04cfde']
    CellCounts(BT_cells, savedir, 'Distribution of cell subsets','BaseDist',pie_color_list)
  
    #Nearest Neighbors in B-T aggs
    btnn=pd.concat(NN,axis=0)
    btnn.to_pickle(savedir+'NearestNeighbors.pkl')
    for i in range(1,9):
        subset=btnn.loc[btnn['Subset_id']==i]
        nn=subset.groupby('neighbor_class').count()['sample']
        nn=nn.reset_index()
        nn.columns=['neighbor_class','count']
        nn['real_class']=[class_dict[x] for x in nn['neighbor_class']]
        fig = plt.figure(figsize=(6,6), frameon=False)
        ax=fig.add_axes([0.15,0.2,0.75,0.75])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False) 
        ax.bar(nn['real_class'],nn['count'],color=c_list)
        plt.ylabel('Cell Counts')
        plt.xticks(rotation=45)
        plt.title('Nearest Neighbors of '+class_dict[i])
        plt.savefig(savedir+class_dict[i]+'_NearestNeighbors.png')
        plt.show()
   
    # Get overall breakdown of T cell subsets
    cd4=BT_cells.loc[BT_cells['class_id']==2]
    cd4=get_allTsubsets(cd4)
    cd4sub=CellSubsets(cd4, savedir, 'Distribution of CD4 T cell subsets','CD4TSecondary_BTaggs')
    cd8=BT_cells.loc[BT_cells['class_id']==3]
    cd8=get_allTsubsets(cd8)
    cd8sub=CellSubsets(cd8, savedir, 'Distribution of CD8 T cell subsets','CD8TSecondary_BTaggs')
    dn=BT_cells.loc[BT_cells['class_id']==1]
    dn=get_allTsubsets(dn)
    dnsub=CellSubsets(dn, savedir, 'Distribution of DN T cell subsets','CD8TSecondary_BTaggs')
    sys.stdout.close()
    sys.stdout=stdoutOrigin
    
if __name__=='__main__':
    main()