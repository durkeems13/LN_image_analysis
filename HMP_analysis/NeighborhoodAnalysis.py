#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:10:32 2021

@author: abrahamr
"""
# Nearest Neighbor analysis
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, pearsonr
from scipy.spatial.distance import euclidean

def CheckMinDist_global(cells,sample):
    nearest_neighbors=[]
    for index,row in cells.iterrows():  
        cell_id=row['cell_id']
        class_id=row['Subset_id']
        d1=[row['global_centroid_column'],row['global_centroid_row']]
        #Check within the size of an ROI
        row_max=d1[1]+512
        row_min=d1[1]-512
        col_max=d1[0]+512
        col_min=d1[0]-512
        group=cells.loc[(cells['global_centroid_column']<col_max)&
                        (cells['global_centroid_column']>col_min)&
                        (cells['global_centroid_row']<row_max)&
                        (cells['global_centroid_row']>row_min)]
   
        #if there are no cells within 512 pixels, broaden to 10000 pixels
        if group.shape[0]==1:
            row_max=d1[1]+10000
            row_min=d1[1]-10000
            col_max=d1[0]+10000
            col_min=d1[0]-10000
            group=cells.loc[(cells['global_centroid_column']<col_max)&
                            (cells['global_centroid_column']>col_min)&
                            (cells['global_centroid_row']<row_max)&
                            (cells['global_centroid_row']>row_min)]
        if group.shape[0]==0: #if still none, then broaden to entire sample
                group=cells
        dists = {}
        for index2,row2 in group.iterrows():
            cell_id_2=row2['cell_id']
            if cell_id != cell_id_2:
                d2=[row2['global_centroid_column'],row2['global_centroid_row']]
                dist=euclidean(d1,d2)
                dists.update({dist:(cell_id_2,row2['Subset_id'])})         
        min_dist = min(dists.keys()) #keeps as pixel
        nearest_neighbors.append(pd.DataFrame([[cell_id,class_id,dists[min_dist][0],dists[min_dist][1],min_dist,sample]],columns=['cell_id','Subset_id','neighbor_id','neighbor_class','min_dist','sample']))         
    nearest_neighbors_df=pd.concat(nearest_neighbors,axis=0)
    return nearest_neighbors_df


def CellCounts(feats, sd, title,savestr,color_list):
    class_dict={1:'DN',2:'CD4+',3:'CD8+',4:'CD20+',5:'CD138+',6:'Treg',7:'Tfh',8:'exhausted_CD8+'}
    c_list=[]
    for ind,row in feats.iterrows():
        c=class_dict[row['class_id']]
        c_list.append(c)
    feats['class_id_real']=c_list
    Cell_type_counts = feats.groupby(['class_id_real']).count()['cell_id']
    Cell_type_counts = Cell_type_counts.reset_index()                                        
    Cell_type_counts = Cell_type_counts.fillna(0)    
    Cell_type_counts.columns=['class_id_real','Counts']    
    Total_cells=sum(Cell_type_counts.Counts)
    Cell_type_counts['Percentage']=(Cell_type_counts['Counts']/Total_cells)*100  
    print(Cell_type_counts)
    textprops={'fontsize':'14','fontname':'Times New Roman'}
    plt.clf()
    fig = plt.figure(figsize=(5,4.75), frameon=False)
    ax=fig.add_axes([0.15,0,0.75,0.75])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.pie(Cell_type_counts['Percentage'], labels=Cell_type_counts['class_id_real'],autopct='%1.1f%%',colors=color_list,textprops=textprops,pctdistance=0.8)              
    plt.title(title,**textprops)
    plt.savefig(sd+savestr+'_ClassBreakdown_prob.png')
    plt.show()
    return Cell_type_counts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvdir", default= "CellFeats_csvs", help = "path to directory with data files (csvs)") 
    parser.add_argument("--svdr",default="NeighborhoodAnalysis/", help = "path to save directory")
    parser.add_argument("--outfile",default="neighborhood_analysis.txt", help = "text file to save outputs to")
    args = parser.parse_args()
    savedir=args.svdr
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    csv_dir=args.csvdir
    outfile=os.path.join(savedir,args.outfile)
    
    stdoutOrigin=sys.stdout
    sys.stdout = open(str(outfile),'w')

    csvs=os.listdir(csv_dir)
    csvs=[x for x in csvs if x.endswith('.csv')]
    biopsy_list=[x.split('_')[0] for x in csvs]
    biopsy_list=list(np.unique(biopsy_list))
    nn_total=[]
    class_dict={1:'DN',2:'CD4+',3:'CD8+',4:'CD20+',5:'CD138+',6:'Treg',7:'Tfh',8:'exhausted_CD8+'}
    c_list=['#0bb81f','#b80b0b','#ba73a6', '#91193d','#0425de','#5f60b3','#ba04de','#04cfde']

    for b in biopsy_list:
        samples=[x for x in csvs if x.startswith(b)==True]
        if len(samples)>1:
            nn=[]
            for c in samples:
                x=('_').join(c.split('_')[:-1])
                cells=pd.read_csv(os.path.join(csv_dir,c))
                nn_section=CheckMinDist_global(cells,x)
                nn.append(nn_section)
            nn=pd.concat(nn,axis=0)
            nn_total.append(nn)
        else:
            x=('_').join(samples[0].split('_')[:-1])
            cells=pd.read_csv(os.path.join(csv_dir,samples[0]))
            nn=CheckMinDist_global(cells,x)
            nn_total.append(nn)
       

    # Nearest Neighbor Graphs
    nn_total_df=pd.concat(nn_total, axis=0)
    nn_total_df.to_pickle(savedir+'NearestNeighbors.pkl')
    for i in range(1,9):
        subset=nn_total_df.loc[nn_total_df['Subset_id']==i]
        N=subset.groupby('neighbor_class').count()['sample']
        N=N.reset_index()
        N.columns=['neighbor_class','count']
        N['real_class']=[class_dict[x] for x in N['neighbor_class']]
        fig = plt.figure(figsize=(6,6), frameon=False)
        ax=fig.add_axes([0.15,0.2,0.75,0.75])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False) 
        ax.bar(N['real_class'],N['count'],color=c_list)
        plt.ylabel('Cell Counts')
        plt.xticks(rotation=45)
        plt.title('Nearest Neighbors of '+class_dict[i])
        plt.savefig(savedir+class_dict[i]+'_NearestNeighbors.png')
        plt.show()

    sys.stdout.close()
    sys.stdout=stdoutOrigin   
    
if __name__=='__main__':
    main()