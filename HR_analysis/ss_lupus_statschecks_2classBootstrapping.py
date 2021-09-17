#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:55:33 2020

@author: rebaabraham
"""

### Lupus single stain cellular analysis
### 2 class bootstrapping analysis 

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import sem, mannwhitneyu, ks_2samp, ttest_ind
from scipy.spatial.distance import euclidean
from math import sqrt
from datetime import datetime

 ## Extract Accession numbers and make new column
def Extract_acc(Case_str):
    sep = '-'
    name_str = Case_str.replace('-','_')
    name_str = name_str.replace('s','S')
    name = name_str.split('_')[0:2] 
    Acc = sep.join(name)
    return Acc

## Compare list of accessions with clinical data, extract cells with matching accessions
def compare_acc(lst1,lst2, df):
    acc = list(set(lst1)&set(lst2))
    return df.loc[(df['Accession #']).isin(acc) == True]               

# Fx for extracting data from clinical spreadsheet, adding to the feature dfs, renaming column
def ExtractfromClin(clin_df,add_df,feat,featname):
    feat_dict = dict(zip(clin_df['Accession #'],clin_df[feat]))
    def check_feat(Acc):
        return feat_dict[Acc]
    add_df[featname] = add_df['Accession #'].apply(check_feat)
    return add_df 

def MakeCellDF(df, clin_dat, prob):    
    cells = pd.read_csv(df)
    cells = cells[(cells.Area<100)&(cells.Area>3)&(cells.Prob>prob)]
    cells['Accession #']=cells['Case'].apply(Extract_acc)
    ## Filters out those that are missing accessions
    cells = cells.loc[(cells['Accession #']).str.startswith('S') == True]
    # Extracts features from biopsies with clinical data
    cells = compare_acc(list(cells['Accession #'].unique()), list(clin_dat['Accession #'].unique()),cells)
    cells = ExtractfromClin(clin_dat,cells,'ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)','ESRD')
    cells.rename(columns={'Unnamed: 0':'index'}, inplace=True)
    print(cells.shape)
    print("Number of biopsies: "+str(len(cells['Accession #'].unique())))
    return cells

## Generate cell counts and ratios per biopsy, and per ROI
    ## Compare ESRD + vs -, make histograms 
    ## look at distribution of cells per ROI in ESRD + vs - 

def CellCounts(feats, Class_list, clin_dat,sd):                                
    Cell_type_counts = feats.groupby(['Accession #','Class_id']).count()
    Cell_type_counts = Cell_type_counts.reset_index()
    Cell_type_counts = Cell_type_counts.pivot(index='Accession #', columns='Class_id',values='Area')
    Cell_type_counts = Cell_type_counts.reset_index()                                          
    Cell_type_counts = Cell_type_counts.fillna(0)  
    Cell_type_counts['total_cells']=sum([Cell_type_counts['CD20+'],Cell_type_counts['CD11c+'],Cell_type_counts['BDCA2+'],Cell_type_counts['CD3+CD4+'],Cell_type_counts['CD3+CD4-']])                       
    for i in Class_list:
        for ii in Class_list:
            if ii == i:
                pass
            else:
                ratio = Cell_type_counts[i]/Cell_type_counts[ii]
                name = str(i+'_to_'+ii) 
                Cell_type_counts[name] = ratio        
    Cell_type_counts = Cell_type_counts.replace([np.inf,np.NaN],[0,0])  
    ExtractfromClin(clin_dat,Cell_type_counts,'ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)','ESRD')
    
    Cell_type_counts.to_csv(sd+'LupusBiopsy_CellCountFeats.csv')
    return Cell_type_counts
            
def CellCountsbyROI(feats,Class_list, clin_dat):  
    Cell_type_byROI = feats.groupby(['Case','Class_id']).count()['Area']
    Cell_type_byROI = Cell_type_byROI.reset_index()
    Cell_type_byROI = Cell_type_byROI.pivot(index='Case', columns='Class_id',values='Area')
    Cell_type_byROI = Cell_type_byROI.reset_index()                                          
    Cell_type_byROI = Cell_type_byROI.fillna(0) 
    Cell_type_byROI['total_cells']=sum([Cell_type_byROI['CD20+'],Cell_type_byROI['CD11c+'],Cell_type_byROI['BDCA2+'],Cell_type_byROI['CD3+CD4+'],Cell_type_byROI['CD3+CD4-']])
    for i in Class_list:
        for ii in Class_list:
            if ii == i:
                pass
            else:
                ratio = Cell_type_byROI[i]/Cell_type_byROI[ii]
                name = str(i+'_to_'+ii) 
                Cell_type_byROI[name] = ratio        
    Cell_type_byROI = Cell_type_byROI.replace([np.inf,np.NaN],[0,0])  
    Cell_type_byROI['Accession #']=Cell_type_byROI['Case'].apply(Extract_acc)
    ExtractfromClin(clin_dat,Cell_type_byROI,'ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)','ESRD')
    return Cell_type_byROI

# Bootstrapping analysis--ROIs are pooled and sampled, sample feature means are calculated, and the difference in means are also calculated 
def bootstrap_meanCI(df,ROI_list,feat,n_iter,sample_n): 
    print(feat)
    ESRD0=df.loc[df['ESRD']==0]
    ESRD1=df.loc[df['ESRD']==1]
    mean0_feat=[]
    mean1_feat=[]
    mean_diff_feat=[]
    for n in range(n_iter):
        ESRD0_rand=list(np.random.choice(ROI_list.loc[ROI_list['ESRD']==0]['Case'],sample_n,replace=True))
        ESRD1_rand=list(np.random.choice(ROI_list.loc[ROI_list['ESRD']==1]['Case'],sample_n,replace=True))
        iter0=[]
        iter1=[]
        for roi in ESRD0_rand:
            val0=ESRD0.loc[ESRD0['Case']==roi].iloc[0][feat]
            iter0.append(val0)
        iter0_mean=sum(iter0)/len(iter0)
        mean0_feat.append(iter0_mean)
        for roi in ESRD1_rand:
            val1=ESRD1.loc[ESRD1['Case']==roi].iloc[0][feat]
            iter1.append(val1)
        iter1_mean=sum(iter1)/len(iter1)
        mean1_feat.append(iter1_mean) 
        iter_diff=iter0_mean-iter1_mean
        mean_diff_feat.append(iter_diff)
    mean0_mean=sum(mean0_feat)/len(mean0_feat)
    mean1_mean=sum(mean1_feat)/len(mean1_feat)
    meandiff=sum(mean_diff_feat)/len(mean_diff_feat)
    CI0=[np.percentile(mean0_feat,2.5),np.percentile(mean0_feat,97.5)]
    CI1=[np.percentile(mean1_feat,2.5),np.percentile(mean1_feat,97.5)]
    CI_diff=[np.percentile(mean_diff_feat,2.5),np.percentile(mean_diff_feat,97.5)]

    if (CI_diff[0]<0) & (CI_diff[1]>0):
        CI_interpretation='overlapping confidence interval'
    elif (CI_diff[0]<0) & (CI_diff[1]<0):
        CI_interpretation='ESRD- less than ESRD+'
    elif (CI_diff[0]>0) & (CI_diff[1]>0):
        CI_interpretation='ESRD- greater than ESRD+'
    d=zip(mean0_feat,mean1_feat,mean_diff_feat)
    v=[mean0_mean,CI0,mean1_mean,CI1,meandiff,CI_diff,CI_interpretation]
    bootstrap=pd.DataFrame(d,columns=['mean0','mean1','diff'])
    final_vals=pd.DataFrame([v],columns=['mean0_mean','CI_0','mean1_mean','CI_1','diff_means','CI_diff','interpretation'])
    return bootstrap, final_vals

    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--df", default= "CellData_forpublication.csv", help = "path to data file (csv)") 
    parser.add_argument("--clin_dat", default="ClinicalData_forpublication.csv", help= "clinical data file")
    parser.add_argument("--svdr",default="HR_Analysis/")
    parser.add_argument("--outfile",default="2class_bootstrapping.txt")
    args = parser.parse_args()
    sd = args.svdr
    if os.path.exists(sd):
        pass
    else:
        os.makedirs(sd)
    ## generates log of output 
    stdoutOrigin=sys.stdout
    sys.stdout = open(str(sd+args.outfile),'w')
    print('Analysis generated on: ')
    print('Data file: '+args.df)
    print(datetime.now())
    print('plots saved in: '+sd)
    print('Prob cutoff: 0.3')
    # Import clinical data 
    clin_dat = pd.read_csv(args.clin_dat)
    clin_dat['Follow up: years'] = clin_dat['Duration of follow up (days)']/365
    clin_dat=clin_dat.loc[clin_dat['Follow up: years']>2]
    print('Restricted to 2 year follow up')
    # import cell features, clean data
    all_cells = MakeCellDF(args.df, clin_dat, 0.3)
    Class_list = ['CD3+CD4+','CD3+CD4-','CD20+','BDCA2+','CD11c+']

### How many ROIs per ESRD+ vs -  
    ROI_list=pd.DataFrame(all_cells.Case.unique(),columns=['Case'])
    ROI_list['Accession #']=ROI_list['Case'].apply(Extract_acc)
    ROI_list= ExtractfromClin(clin_dat,ROI_list,'ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)','ESRD')
    
    print('# ESRD + vs - ROIs')
    print(ROI_list.groupby('ESRD').count())
    Cell_type_countsbyROI = CellCountsbyROI(all_cells,Class_list, clin_dat)
    feat_list=['CD3+CD4+','CD3+CD4-','CD20+','BDCA2+','CD11c+']
### Randomly sampling from ROIs in each category
    if not os.path.exists(os.path.join(sd,'Bootstrap_byROI')):
        os.makedirs(os.path.join(sd,'Bootstrap_byROI'))

    Acc_list=ROI_list.groupby('Accession #').count()['Case']
    Acc_list=Acc_list.reset_index()
    Acc_list= ExtractfromClin(clin_dat,Acc_list,'ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)','ESRD')

    n_iter=1000
    sample_n=200
    print('Bootstrap by ROI, sample_n=200')
    for feat in feat_list:
        feat_bs,interp=bootstrap_meanCI(Cell_type_countsbyROI,ROI_list,feat,n_iter,sample_n)
        textargs={'size':'16','fontname':'Times New Roman'}
        plt.clf()
        fig=plt.figure(figsize=(5.5,5),frameon=False)
        ax=fig.add_axes([0.15,0.15,0.75,0.75])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.hist([feat_bs['mean0'],feat_bs['mean1']],color=['blue','red'], bins=50)
        plt.xticks(fontsize=14,fontname='Times New Roman')
        plt.yticks(fontsize=14,fontname='Times New Roman')
        plt.ylabel('Number of Samples',**textargs)
        plt.xlabel('Mean of '+feat+' per ROI',**textargs)
        plt.savefig(sd+'Bootstrap_byROI/'+'Means_'+feat+'.png')
        plt.show()
       
        plt.clf()
        fig=plt.figure(figsize=(5.5,5),frameon=False)
        ax=fig.add_axes([0.15,0.15,0.75,0.75])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.hist(feat_bs['diff'],color=['purple'], bins=50)
        plt.xticks(fontsize=14,fontname='Times New Roman')
        plt.yticks(fontsize=14,fontname='Times New Roman')
        plt.ylabel('Number of Samples',**textargs)
        plt.xlabel('Difference in mean '+feat+' per ROI',**textargs)
        plt.savefig(sd+'Bootstrap_byROI/'+'Diff_means_'+feat+'.png')
        plt.show()
        print('Avg mean '+feat+'for ESRD-: '+str(feat_bs['mean0'].mean())+' '+str(interp['CI_0']))
        print('Avg mean '+feat+'for ESRD+: '+str(feat_bs['mean1'].mean())+' '+str(interp['CI_1']))
        print('Avg mean '+feat+'for diff means: '+str(feat_bs['diff'].mean())+' '+str(interp['CI_diff']))
        print(interp['interpretation'][0])
    
    sys.stdout.close()
    sys.stdout=stdoutOrigin 
    
if __name__=='__main__':
    main()
    
