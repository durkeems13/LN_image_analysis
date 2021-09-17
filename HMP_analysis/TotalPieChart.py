#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 09:37:00 2021

@author: abrahamr
"""
#Code for generating basic pie charts of cellular distributions; these functions get used in other scripts
import os, sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

def get_full_biopsy_df(csv_dir,csvs):
    biopsy_list=[x.split('_')[0] for x in csvs]
    biopsy_list=list(np.unique(biopsy_list))
    every_cell=[]
    for csv in csvs:
        b=csv.split('_')[0]
        sample='_'.join(csv.split('_')[:-1])
        df=pd.read_csv(os.path.join(csv_dir,csv))
        df['Accession #']=b
        df['sample']=sample
        every_cell.append(df)
    every_cell_df=pd.concat(every_cell, axis=0)
    return every_cell_df

def CellCounts(feats, sd, title,savestr,color_list):
    class_dict={1:'DN',2:'CD4+',3:'CD8+',4:'CD20+',5:'CD138+'}
    Cell_type_counts = feats.groupby(['class_id']).count()['cell_id']
    Cell_type_counts = Cell_type_counts.reset_index()                                        
    Cell_type_counts = Cell_type_counts.fillna(0)    
    Cell_type_counts.columns=['class_id','Counts'] 
    Cell_type_counts['class_real_id']=[class_dict[x] for x in Cell_type_counts['class_id']]   
    Total_cells=sum(Cell_type_counts.Counts)
    Cell_type_counts['Percentage']=(Cell_type_counts['Counts']/Total_cells)*100  
    print(Cell_type_counts)
    textprops={'fontsize':'16','fontname':'Times New Roman','fontweight':'bold'}
    plt.clf()
    fig = plt.figure(figsize=(5,4.75), frameon=False)
    ax=fig.add_axes([0.1,0,0.75,0.75])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.pie(Cell_type_counts['Percentage'], labels=None,autopct=None,colors=color_list,textprops=textprops,pctdistance=0.75)              
    plt.title(title,**textprops)
    plt.savefig(sd+savestr+'_ClassBreakdown_prob.png')
    plt.show()
    return Cell_type_counts

def CellCounts_subset(feats, sd, title,savestr,color_list):
    class_dict={1:'DN',2:'CD4+',3:'CD8+',4:'CD20+',5:'CD138+',6:'Treg',7:'Tfh',8:'exhausted_CD8+'}
    Cell_type_counts = feats.groupby(['Subset_id']).count()['cell_id']
    Cell_type_counts = Cell_type_counts.reset_index()                                        
    Cell_type_counts = Cell_type_counts.fillna(0)    
    Cell_type_counts.columns=['Subset_id','Counts'] 
    Cell_type_counts['Subset_real_id']=[class_dict[x] for x in Cell_type_counts['Subset_id']]
    Total_cells=sum(Cell_type_counts.Counts)
    Cell_type_counts['Percentage']=(Cell_type_counts['Counts']/Total_cells)*100  
    print(Cell_type_counts)
    explode=(0,0,0,0,0,0.3,0.3,0.3)
    textprops={'fontsize':'10','fontname':'Times New Roman'}
    plt.clf()
    fig = plt.figure(figsize=(5,4.75), frameon=False)
    ax=fig.add_axes([0.15,0,0.75,0.75])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.pie(Cell_type_counts['Percentage'], labels=Cell_type_counts['Subset_real_id'],autopct='%1.1f%%',colors=color_list,textprops=textprops,pctdistance=0.85, explode=explode)              
    plt.title(title,**textprops)
    plt.savefig(sd+savestr+'_ClassBreakdown_prob.png')
    plt.show()
    return Cell_type_counts


def main():
    savedir='PieCharts'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    csv_dir='CellFeats_csvs_20210616'
    csvs=os.listdir(csv_dir)
    csvs=[x for x in csvs if x.endswith('.csv')]
    cells=get_full_biopsy_df(csv_dir,csvs)
 #Make T cell pie chart;
    tcells=cells.loc[cells['class_id']<=3]
    CellCounts(tcells, savedir, '','TDist',['#0bb81f','#b80b0b','#0425de'])
#Make all cell pie chart:
    color_list=['#0bb81f','#b80b0b','#0425de', '#ba04de','#04cfde']
    CellCounts(cells, savedir, '','BaseDist',color_list)

    sub_color_list=['#0bb81f','#b80b0b','#0425de','#ba04de','#04cfde','#ba73a6','#91193d','#5f60b3',]
    CellCounts_subset(cells, savedir, 'Distribution of cell subsets','SubsetDist',sub_color_list)

if __name__ == '__main__':
    main()