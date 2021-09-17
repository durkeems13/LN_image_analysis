#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:55:33 2020

@author: rebaabraham
"""

### Lupus single stain cellular analysis
### 3 class bootstrapping analysis 

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



def threeclassbootstrap_meanCI(df,ROI_list,feat,n_iter,sample_n): 
    print(feat)
    ESRD0=df.loc[df['ESRD']==0]
    ESRD1=df.loc[df['ESRD']==1]
    ESRD2=df.loc[df['ESRD']==2]
    mean0_feat=[]
    mean1_feat=[]
    mean2_feat=[]
    diff01=[]
    diff02=[]
    diff12=[]
    for n in range(n_iter):
        ESRD0_rand=list(np.random.choice(ROI_list.loc[ROI_list['ESRD']==0]['Case'],sample_n,replace=True))
        ESRD1_rand=list(np.random.choice(ROI_list.loc[ROI_list['ESRD']==1]['Case'],sample_n,replace=True))
        ESRD2_rand=list(np.random.choice(ROI_list.loc[ROI_list['ESRD']==2]['Case'],sample_n,replace=True))
        iter0=[]
        iter1=[]
        iter2=[]
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
        for roi in ESRD2_rand:
            val2=ESRD2.loc[ESRD2['Case']==roi].iloc[0][feat]
            iter2.append(val2)
        iter2_mean=sum(iter2)/len(iter2)
        mean2_feat.append(iter2_mean)
        iter01=iter0_mean-iter1_mean
        diff01.append(iter01)
        iter02=iter0_mean-iter2_mean
        diff02.append(iter02)
        iter12=iter1_mean-iter2_mean
        diff12.append(iter12)
    mean0_mean=sum(mean0_feat)/len(mean0_feat)
    mean1_mean=sum(mean1_feat)/len(mean1_feat)
    mean2_mean=sum(mean2_feat)/len(mean2_feat)
    CI0=[np.percentile(mean0_feat,2.5),np.percentile(mean0_feat,97.5)]
    CI1=[np.percentile(mean1_feat,2.5),np.percentile(mean1_feat,97.5)]
    CI2=[np.percentile(mean2_feat,2.5),np.percentile(mean2_feat,97.5)]   
    CId01=[np.percentile(diff01,2.5),np.percentile(diff01,97.5)]
    CId02=[np.percentile(diff02,2.5),np.percentile(diff02,97.5)]
    CId12=[np.percentile(diff12,2.5),np.percentile(diff12,97.5)]
    d=zip(mean0_feat,mean1_feat, mean2_feat,diff01,diff02,diff12)
    v=[mean0_mean,CI0,mean1_mean,CI1,mean2_mean,CI2,CId01,CId02,CId12]
    bootstrap=pd.DataFrame(d,columns=['mean0','mean1','mean2','d01','d02','d12'])
    final_vals=pd.DataFrame([v],columns=['mean0_mean','CI_0','mean1_mean','CI_1','mean2_mean','CI_2','CI_d01','CI_d02','CI_d12'])
    return bootstrap, final_vals


  
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--df", default= "CellData_forpublication.csv", help = "path to data file (csv)") 
    parser.add_argument("--clin_dat", default="ClinicalData_forpublication.csv", help= "clinical data file")
    parser.add_argument("--svdr",default="HR_Analysis/")
    parser.add_argument("--outfile",default="3class_bootstrapping.txt")
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
    clin_dat.loc[clin_dat['Time to ESRD (days)']<14,'ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)']=2
    print('Restricted to 2 year follow up')
    # import cell features, clean data
    all_cells = MakeCellDF(args.df, clin_dat, 0.3)
### How many ROIs per ESRD+ vs -  
    ROI_list=pd.DataFrame(all_cells.Case.unique(),columns=['Case'])
    ROI_list['Accession #']=ROI_list['Case'].apply(Extract_acc)
    ROI_list= ExtractfromClin(clin_dat,ROI_list,'ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)','ESRD')
    
    print('# ESRD + vs - ROIs')
    print(ROI_list.groupby('ESRD').count())
    feat_list=['CD3+CD4+','CD3+CD4-','CD20+','BDCA2+','CD11c+']
    Cell_type_countsbyROI = CellCountsbyROI(all_cells,feat_list, clin_dat)
    

### Randomly sampling from ROIs in each category
    print('Bootstrap by ROI')

    if os.path.exists(os.path.join(sd,'Bootstrap_byROI_3class'))==True:
        pass
    else:
        os.makedirs(os.path.join(sd,'Bootstrap_byROI_3class'))

    Acc_list=ROI_list.groupby('Accession #').count()['Case']
    Acc_list=Acc_list.reset_index()
    Acc_list= ExtractfromClin(clin_dat,Acc_list,'ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)','ESRD')

    n_iter=1000
    sample_n=150
    for feat in feat_list:
        feat_bs,interp=threeclassbootstrap_meanCI(Cell_type_countsbyROI,ROI_list,feat,n_iter,sample_n)
        textargs={'size':'16','fontname':'Times New Roman'}
        plt.clf()
        fig=plt.figure(figsize=(5.5,5),frameon=False)
        ax=fig.add_axes([0.15,0.15,0.75,0.75])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.hist([feat_bs['mean0'],feat_bs['mean1'], feat_bs['mean2']],color=['blue','#fc9003','green'], bins=50)
        plt.xticks(fontsize=14,fontname='Times New Roman')
        plt.yticks(fontsize=14,fontname='Times New Roman')
        plt.ylabel('Number of Samples',**textargs)
        plt.xlabel('Mean '+feat+' per ROI',**textargs)
        plt.savefig(sd+'Bootstrap_byROI_3class/'+'Means_'+feat+'.png')
        plt.show()
        
        plt.clf()
        fig=plt.figure(figsize=(5.5,5),frameon=False)
        ax=fig.add_axes([0.15,0.15,0.75,0.75])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.hist([feat_bs['d01'],feat_bs['d02'],feat_bs['d12']],color=['purple','cyan','yellow'], bins=50)
        plt.xticks(fontsize=14,fontname='Times New Roman')
        plt.yticks(fontsize=14,fontname='Times New Roman')
        plt.ylabel('Number of Samples',**textargs)
        plt.xlabel('Difference in mean '+feat+' per ROI',**textargs)
        plt.savefig(sd+'Bootstrap_byROI_3class/'+'Diff_means_'+feat+'.png')
        plt.show()
        print('Avg mean '+feat+'for ESRD-: '+str(feat_bs['mean0'].mean())+' '+str(interp['CI_0']))
        print('Avg mean '+feat+'for ESRD+: '+str(feat_bs['mean1'].mean())+' '+str(interp['CI_1']))
        print('Avg mean '+feat+'for ESRDcurrent: '+str(feat_bs['mean2'].mean())+' '+str(interp['CI_2']))
        print('Avg diff means '+feat+'for ESRD- vs ESRD+: '+str(feat_bs['d01'].mean())+' '+str(interp['CI_d01']))
        print('Avg diff means '+feat+'for ESRD- vs ESRDcurrent: '+str(feat_bs['d02'].mean())+' '+str(interp['CI_d02']))
        print('Avg diff means '+feat+'for ESRD+ vs ESRDcurrent: '+str(feat_bs['d12'].mean())+' '+str(interp['CI_d12']))
   
    sys.stdout.close()
    sys.stdout=stdoutOrigin 
    
if __name__=='__main__':
    main()
    