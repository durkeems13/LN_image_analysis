#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:12:19 2021

@author: abrahamr
"""
# Digging into secondary markers 
import os, sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime


def get_full_biopsy_df(csv_dir,csvs):
    biopsy_list=[x.split('_')[0] for x in csvs]
    biopsy_list=list(np.unique(biopsy_list))
    every_cell=[]
    for b in biopsy_list:
        samples=[x for x in csvs if x.startswith(b)==True]
        if len(samples)>1:
            for biopsy in samples:
                sample='_'.join(biopsy.split('_')[:-1])
                df=pd.read_csv(os.path.join(csv_dir,biopsy))
                df['Accession #']=b
                df['sample']=sample
                every_cell.append(df)
        else:
            sample='_'.join(biopsy.split('_')[:-1])
            df=pd.read_csv(os.path.join(csv_dir,samples[0]))
            df['Accession #']=b
            df['sample']=sample
            every_cell.append(df)
    every_cell_df=pd.concat(every_cell, axis=0)
    return every_cell_df

def get_allTsubsets(df):
    sub_dict={(0,0,0):'PD1-ICOS-FoxP3-',(1,0,0):'PD1+ICOS-FoxP3-',
              (1,1,0):'PD1+ICOS+FoxP3-',(1,1,1):'PD1+ICOS+FoxP3+',
              (0,1,0):'PD1-ICOS+FoxP3-',(0,1,1):'PD1-ICOS+FoxP3+',
              (1,0,1):'PD1+ICOS-FoxP3+',(0,0,1):'PD1-ICOS-FoxP3+'} 
    subset_ids=[]
    for ind,cell in df.iterrows():
        check=(cell['pd1_expression'],cell['icos_expression'],cell['foxp3_expression'])
        if check in sub_dict.keys():
            subset_ids.append(sub_dict[check])
        else:
             subset_ids.append(cell['class_id'])
    df['Secondary_exp']=subset_ids
    return df

def CellSubsets(feats, sd, title,savestr):
    Cell_type_counts = feats.groupby(['Secondary_exp']).count()['cell_id']
    Cell_type_counts = Cell_type_counts.reset_index()                                        
    Cell_type_counts = Cell_type_counts.fillna(0)    
    Cell_type_counts.columns=['Secondary_exp','Counts']    
    Total_cells=sum(Cell_type_counts.Counts)
    Cell_type_counts['Percentage']=(Cell_type_counts['Counts']/Total_cells)*100  
    Cell_type_counts=Cell_type_counts.sort_values('Percentage')
    if Cell_type_counts.shape[0]==8:
        explode=(0.3,0.3,0.3,0.3,0,0,0,0)
    else:
        explode=np.repeat(0,Cell_type_counts.shape[0])
    color_list=['#8B008B','#008B8B','#A9A9A9','#556B2F','#9932CC','#A9A9A9','#483D8B']
    print(Cell_type_counts)
    textprops={'fontsize':'12','fontname':'Times New Roman'}
    plt.clf()
    fig = plt.figure(figsize=(7,5), frameon=False)
    ax=fig.add_axes([0.1,0,0.75,0.75])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.pie(Cell_type_counts['Percentage'], labels=Cell_type_counts['Secondary_exp'],
           autopct=None,explode=explode,colors=color_list,textprops=textprops,pctdistance=0.8)              
    plt.title(title,**textprops)
    plt.savefig(sd+savestr+'_ClassBreakdown_prob.png')
    plt.show()
    return Cell_type_counts

def ExtractfromClin(clin_df,add_df,feat,featname): #pulls values from clinical data by Accession # and adds to another data frame
    feat_dict = dict(zip(clin_df['Accession #'],clin_df[feat]))
    def check_feat(Acc):
        return feat_dict[Acc]
    add_df[featname] = add_df['Accession #'].apply(check_feat)
    return add_df 

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvdir", default= "CellFeats_csvs", help = "path to data file (csv)") 
    parser.add_argument("--svdr",default='AnalyzeSecondaries/', help = "path to save directory")
    parser.add_argument("--outfile",default="secondaries.txt", help = "text file to save outputs to")
    args = parser.parse_args()
    savedir=args.svdr
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    csv_dir=args.csvdir
    outfile=savedir+args.outfile
    stdoutOrigin=sys.stdout
    sys.stdout = open(str(outfile),'w')
    print('Analysis generated on: ')
    print('Data folder: '+csv_dir)
    print(datetime.now())
    print('plots saved in: '+savedir)
    csvs=os.listdir(csv_dir)
    csvs=[x for x in csvs if x.endswith('.csv')]
    cells=get_full_biopsy_df(csv_dir,csvs)
    dn=cells.loc[cells['class_id']==1]
    dn=get_allTsubsets(dn)
    dnsub=CellSubsets(dn, savedir, 'Distribution of DN T cell subsets','DNTSecondary')
    cd4=cells.loc[cells['class_id']==2]
    cd4=get_allTsubsets(cd4)
    cd4sub=CellSubsets(cd4, savedir, 'Distribution of CD4 T cell subsets','CD4TSecondary')
    cd8=cells.loc[cells['class_id']==3]
    cd8=get_allTsubsets(cd8)
    cd8sub=CellSubsets(cd8, savedir, 'Distribution of CD8 T cell subsets','CD8TSecondary')
    
    sys.stdout.close()
    sys.stdout=stdoutOrigin
    
if __name__=='__main__':
    main()